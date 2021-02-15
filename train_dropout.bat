python ./common/train.py      ^
    --model_config            arch6_dropout ^
    --batch_size              256  ^
    --training_schedule       (200,0.001)  (100,0.0005)  (100,0.0002) (100,0.0001)



    @REM --results_filename        arch3_dropout_results_test 
    @REM --dry-run                      ^
