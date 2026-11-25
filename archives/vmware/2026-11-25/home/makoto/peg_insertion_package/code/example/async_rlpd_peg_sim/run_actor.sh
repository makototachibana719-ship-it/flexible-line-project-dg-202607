export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" --actor --debug --env=AuboPegInsert-State-v0 --exp_name=serl_peg_insert_rlpd20_v3_10k_utd1 --seed=0 --max_steps=10000 --random_steps=0
