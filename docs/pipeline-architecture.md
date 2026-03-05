# Pipeline Architecture

Full data flow for the MRI event sequence generation pipeline.

```mermaid
flowchart TD
    subgraph INPUTS["Input Data"]
        RAW["Raw Event CSVs\n40 Scanner Files\nPXChange_Refactored/data/"]
        ARC["Archive Examination CSVs\n175832.csv and 176625.csv\n_archive/SeqofSeq_Pipeline_v1/data/\n1 row = 1 complete MRI scan\n+ duration column by BodyGroup"]
    end

    subgraph PREP["Preprocessing  [Steps 1 and 1b]"]
        PP["preprocessing.py\nExtract exchange sequences\n  pre/post-scan events\nExtract examination sequences\n  in-scan events\nPhase type labeling\n  startup / between / shutdown\ntimediff to inter-event durations"]
        OPP["orchestration_preprocessing.py\nDay-level patient schedules\nBREAK token insertion\n  gaps over 1 hour"]
        DUR["archive_duration_priors.py  NEW\nGroup by BodyGroup\nCompute mean and std of duration\nBuild statistical prior dict"]
    end

    subgraph TRAINING["Model Training  [Steps 2, 3, 2c]"]
        TRX["train_exchange.py\nSequenceGeneratorModel\nbody_from to body_to + phase_type\nd_model=256, 6+6 layers\nLoss: token + 0.5x duration  UPDATED\n+ Duration jitter augmentation  NEW"]
        TREX["train_examination.py\nSequenceGeneratorModel\nbody_region single\nd_model=256, 6+6 layers\nLoss: token + 0.5x duration  UPDATED\n+ Prior regularization loss  NEW\n+ Duration jitter augmentation  NEW"]
        TORC["train_orchestration.py\nOrchestrationModel\nday features + scanner embedding\nd_model=128, 3+4 layers\nLoss: token only"]
    end

    subgraph MODELS["Trained Models"]
        EMD["exchange_model_best.pt"]
        XMD["examination_model_best.pt"]
        OMD["orchestration_model_best.pt"]
    end

    subgraph SIM["Day Simulation  [Steps 4 and 4b]"]
        direction TB
        ORCH_GEN["Orchestrator\nBody region sequence for day\ne.g. BRAIN BREAK SPINE END"]
        EX_FRAME["Exchange Model\nPhase: startup\nbody_from=START to body_to\nOutput: pre-scan setup events + durations"]
        EXAM_CONTENT["Examination Model\nbody_region\nFills scan between tokens 100 and 104\nOutput: in-scan events + durations"]
        EX_BETWEEN["Exchange Model\nPhase: between or shutdown\nOutput: post-scan handoff events + durations"]
        DAY_OUT["Synthetic Day Schedule\nFull timestamped event log\nExchange + Examination interleaved"]
    end

    RAW --> PP
    RAW --> OPP
    ARC --> DUR
    PP -->|exchange sequences| TRX
    PP -->|examination sequences| TREX
    DUR -->|BodyGroup duration priors| TREX
    OPP --> TORC
    TRX --> EMD
    TREX --> XMD
    TORC --> OMD
    OMD --> ORCH_GEN
    ORCH_GEN -->|Body region + BREAK sequence| EX_FRAME
    EMD --> EX_FRAME
    EX_FRAME -->|Token 100 scan boundary handoff| EXAM_CONTENT
    XMD --> EXAM_CONTENT
    EXAM_CONTENT -->|Token 104 scan end return to exchange| EX_BETWEEN
    EMD --> EX_BETWEEN
    EX_BETWEEN --> DAY_OUT
    EXAM_CONTENT --> DAY_OUT
```

## Model Interlock

The Exchange and Examination models are trained separately but interlock seamlessly during simulation:

- **Token 100** (`MRI_MSR_100` = Start Prepare): Exchange emits this as the final token before handing off to the Examination model
- **Token 104** (`MRI_MSR_104` = Measurement Finished OK): Examination emits this as its terminal token, returning control to the Exchange model
- The Day Simulator watches for these boundary tokens and switches active models accordingly
