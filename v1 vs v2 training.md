PS C:\Users\arnav\Downloads\AI-TRAINING> Remove-Item -Force tests\smoke_results\dryrun_v2.json -ErrorAction SilentlyContinue; ./.venv/Scripts/python.exe -m training.smoke_test --arch v1 --steps 3000 ; ./.venv/Scripts/python.exe -m training.smoke_test --arch v2 --steps 3000
=== SMOKE TEST V1 | race ===
device cuda | params 80,055,040 | steps 3000 | peak lr 0.0003
pre_norm=False use_rope=False scale_residual_init=False
Step 0/3000 | Loss 10.4295 | lr 6.00e-07
Step 0/3000 | Validation 10.3948
Step 100/3000 | Loss 9.0196 | lr 6.06e-05
Step 200/3000 | Loss 7.5604 | lr 1.21e-04
Step 250/3000 | Validation 7.3131
Step 300/3000 | Loss 7.1923 | lr 1.81e-04
Step 400/3000 | Loss 6.5988 | lr 2.41e-04
Step 500/3000 | Loss 6.5666 | lr 3.00e-04
Step 500/3000 | Validation 6.3098
Step 600/3000 | Loss 6.2378 | lr 2.99e-04
Step 700/3000 | Loss 5.6258 | lr 2.96e-04
Step 750/3000 | Validation 5.8693
Step 800/3000 | Loss 5.9695 | lr 2.91e-04
Step 900/3000 | Loss 5.7384 | lr 2.83e-04
Step 1000/3000 | Loss 5.5015 | lr 2.74e-04
Step 1000/3000 | Validation 5.6014
Step 1100/3000 | Loss 5.3633 | lr 2.63e-04
Step 1200/3000 | Loss 5.4360 | lr 2.51e-04
Step 1250/3000 | Validation 5.3758
Step 1300/3000 | Loss 5.4835 | lr 2.37e-04
Step 1400/3000 | Loss 5.2239 | lr 2.22e-04
Step 1500/3000 | Loss 5.3077 | lr 2.07e-04
Step 1500/3000 | Validation 5.2105
Step 1600/3000 | Loss 5.2997 | lr 1.90e-04
Step 1700/3000 | Loss 5.2299 | lr 1.73e-04
Step 1750/3000 | Validation 5.0537
Step 1800/3000 | Loss 4.8922 | lr 1.57e-04
Step 1900/3000 | Loss 4.9144 | lr 1.40e-04
Step 2000/3000 | Loss 4.8948 | lr 1.23e-04
Step 2000/3000 | Validation 4.9532
Step 2100/3000 | Loss 4.8431 | lr 1.08e-04
Step 2200/3000 | Loss 5.0207 | lr 9.27e-05
Step 2250/3000 | Validation 4.8627
Step 2300/3000 | Loss 5.0192 | lr 7.89e-05
Step 2400/3000 | Loss 5.0558 | lr 6.66e-05
Step 2500/3000 | Loss 4.7952 | lr 5.58e-05
Step 2500/3000 | Validation 4.8056
Step 2600/3000 | Loss 4.6489 | lr 4.67e-05
Step 2700/3000 | Loss 4.8543 | lr 3.95e-05
Step 2750/3000 | Validation 4.7666
Step 2800/3000 | Loss 4.8127 | lr 3.42e-05
Step 2900/3000 | Loss 4.6736 | lr 3.11e-05
Step 2999/3000 | Validation 4.7368

best 4.7368 | final 4.7368 | spikes 0 | 15,145 tok/s | 40.6 min
wrote tests\smoke_results\race_v1.json
=== SMOKE TEST V2 | race ===
device cuda | params 79,563,520 | steps 3000 | peak lr 0.0003
pre_norm=True use_rope=True scale_residual_init=True
Step 0/3000 | Loss 10.4379 | lr 6.00e-07
Step 0/3000 | Validation 10.4290
Step 100/3000 | Loss 8.8637 | lr 6.06e-05
Step 200/3000 | Loss 7.4097 | lr 1.21e-04
Step 250/3000 | Validation 7.1374
Step 300/3000 | Loss 7.0896 | lr 1.81e-04
Step 400/3000 | Loss 6.3588 | lr 2.41e-04
Step 500/3000 | Loss 6.3206 | lr 3.00e-04
Step 500/3000 | Validation 6.0941
Step 600/3000 | Loss 5.9629 | lr 2.99e-04
Step 700/3000 | Loss 5.3115 | lr 2.96e-04
Step 750/3000 | Validation 5.5607
Step 800/3000 | Loss 5.6435 | lr 2.91e-04
Step 900/3000 | Loss 5.3959 | lr 2.83e-04
Step 1000/3000 | Loss 5.1199 | lr 2.74e-04
Step 1000/3000 | Validation 5.2249
Step 1100/3000 | Loss 4.9994 | lr 2.63e-04
Step 1200/3000 | Loss 5.1041 | lr 2.51e-04
Step 1250/3000 | Validation 4.9946
Step 1300/3000 | Loss 5.1374 | lr 2.37e-04
Step 1400/3000 | Loss 4.8089 | lr 2.22e-04
Step 1500/3000 | Loss 4.9656 | lr 2.07e-04
Step 1500/3000 | Validation 4.8396
Step 1600/3000 | Loss 4.9564 | lr 1.90e-04
Step 1700/3000 | Loss 4.8952 | lr 1.73e-04
Step 1750/3000 | Validation 4.7017
Step 1800/3000 | Loss 4.5262 | lr 1.57e-04
Step 1900/3000 | Loss 4.5739 | lr 1.40e-04
Step 2000/3000 | Loss 4.5384 | lr 1.23e-04
Step 2000/3000 | Validation 4.6043
Step 2100/3000 | Loss 4.4978 | lr 1.08e-04
Step 2200/3000 | Loss 4.6826 | lr 9.27e-05
Step 2250/3000 | Validation 4.5227
Step 2300/3000 | Loss 4.6870 | lr 7.89e-05
Step 2400/3000 | Loss 4.7314 | lr 6.66e-05
Step 2500/3000 | Loss 4.4506 | lr 5.58e-05
Step 2500/3000 | Validation 4.4649
Step 2600/3000 | Loss 4.2738 | lr 4.67e-05
Step 2700/3000 | Loss 4.4977 | lr 3.95e-05
Step 2750/3000 | Validation 4.4241
Step 2800/3000 | Loss 4.4333 | lr 3.42e-05
Step 2900/3000 | Loss 4.3318 | lr 3.11e-05
Step 2999/3000 | Validation 4.3947

best 4.3947 | final 4.3947 | spikes 0 | 14,738 tok/s | 41.7 min
wrote tests\smoke_results\race_v2.json
PS C:\Users\arnav\Downloads\AI-TRAINING> 