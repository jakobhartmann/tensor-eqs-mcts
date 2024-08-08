docker run --gpus all --pid=host --net=host \
--name tensat_eqs_mcts \
-it \
--mount type=bind,source="/home/user/tensor-eqs-mcts/tensat",target=/usr/tensat \
--mount type=bind,source="/home/user/tensor-eqs-mcts/egg",target=/usr/egg \
--mount type=bind,source="/home/user/tensor-eqs-mcts/taso",target=/usr/TASO \
--mount type=bind,source="/home/user/tensor-eqs-mcts/rmcts",target=/usr/rmcts \
--mount type=bind,source="/home/user/tensor-eqs-mcts/experiments",target=/usr/experiments \
tensat:1.0 bash
