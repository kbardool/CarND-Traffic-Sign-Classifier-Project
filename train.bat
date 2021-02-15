python ./common/train.py      ^
    --model_config            arch7 ^
    --batch_size              256 128 64 32 ^
    --training_schedule       (200,0.001)  (100,0.0005)  (100,0.0002) (100,0.0001)
    @REM --results_filename        _results_test 

