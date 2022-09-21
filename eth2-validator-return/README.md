
# eth2-validator-return
Modelling the realised extractable value and use it to dynamically predict annual validator return.

## Requirements:
* python, pip, and [jupyter-notebook](https://jupyter.readthedocs.io/en/latest/install/notebook-classic.html)


## Notebooks

### [Modelling REV using classification and regression tree](eth2-validator-return/rev_modeling_DT_RF.ipynb)
This code shows the models built to estimate REV using Decision Tree Classifier and Random Forest Regressor. The prediction results are output to [rev_randomforest.csv](eth2-validator-return/Data/rev_randomforest.csv) and used in the [validator return notebook](eth2-validator-return/eth2_mev_update.ipynb).

### [Modelling REV using classification and regression tree](eth2-validator-return/rev_modeling_ARIMA.ipynb)
This code shows the ARIMA models attempted and the final model along with its estimated REV. The prediction results are output to [rev_arima.csv](eth2-validator-return/Data/rev_arima.csv) and used in the [validator return notebook](eth2-validator-return/eth2_mev_update.ipynb).


### [Modelling REV using classification and regression tree](eth2-validator-return/eth2_mev_update.ipynb)

This analysis is an update of the previous [article](https://hackmd.io/@flashbots/mev-in-eth2) with additional considerations for the following:
- ETH burned post EIP-1559
- MEV per block update
- Number of validators update
- Participation rate range update

Four types of scenario simulation are conducted:
1. CL reward scenario
  - Attest source, target, head successfully
  - Attest source & target successfully, miss head
  - Attest source successfully, miss head & target
2. Expected number of blocks proposed per year per validator scenario
  - Luckiest 1% from binomial distribution
  - Median from binomial distribution
  - Unluckiest 1% from binomial distribution
3. MEV scenario
This is the analysis of annual validator return using the predicted REV from the previous two notebooks 
  - Random Forest Regressor with confidence intervals [2.5%, 97.5%]
  - ARIMA (2,0,1) with confidence intervals [2.5%, 97.5%]
4. Combined scenario
 - Best scenario combines the following:
    - Attest target, source, head successfully
    - Propose number of blocks at the luckiest 1% level
    - Generate MEV at the highest 1% level
 - worst scenario combines the following:
    - Luckiest 1% of proposer opportunity and top 2.5% model predicted REV
    - Unluckiest 1% of proposer opportunity and bottom 2.5% model predicted REV
    - Average proposer opportunity and model predicted REV

## Articles
### [Modelling Realised Extractable Value in Proof of Stake Ethereum](https://collective.flashbots.net/t/modelling-realised-extractable-value-in-proof-of-stake-ethereum/290)

## Videos
* [Modeling REV in Proof of Stake Ethereum at MEV-SBC Workshop](https://www.youtube.com/watch?v=THKbs5YBWpk)

## Slides
* [Slides presented at the MEV-SBC Workshop](https://docs.google.com/presentation/d/17BOliUgcDQ61GyXEOLZZc8QjDcaixNTsFeuuY1IF7uw/edit#slide=id.g14833007f45_2_56)

