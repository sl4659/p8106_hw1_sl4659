Homework1
================
Shenglin Liu
2/26/2020

# Part a

``` r
library(tidyverse)
```

    ## ── Attaching packages ────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.2.1     ✔ purrr   0.3.3
    ## ✔ tibble  2.1.3     ✔ dplyr   0.8.3
    ## ✔ tidyr   1.0.0     ✔ stringr 1.4.0
    ## ✔ readr   1.3.1     ✔ forcats 0.4.0

    ## ── Conflicts ───────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(ModelMetrics)
```

    ## 
    ## Attaching package: 'ModelMetrics'

    ## The following object is masked from 'package:base':
    ## 
    ##     kappa

``` r
# load data
test = read_csv("./data/solubility_test.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double()
    ## )

    ## See spec(...) for full column specifications.

``` r
train = read_csv("./data/solubility_train.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double()
    ## )
    ## See spec(...) for full column specifications.

``` r
# fit a linear model using least squares
fit_ml = lm(Solubility ~ .-Solubility, data = train)
# summary(fit_ml)
# calculate the mse using the test data
pred_ml  = predict(fit_ml, test)
mse_ml = mse(test$Solubility, pred_ml)
mse_ml
```

    ## [1] 0.5558898

# Part b

``` r
library(ISLR)
library(glmnet)
```

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## Loaded glmnet 3.0-1

``` r
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following objects are masked from 'package:ModelMetrics':
    ## 
    ##     confusionMatrix, precision, recall, sensitivity, specificity

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
# fit the ridge regression (alpha = 0) with a sequence of lambdas
x.train = as.matrix(subset(train, select = -Solubility))
y.train = train$Solubility
# cross-validation
set.seed(1)
cv.ridge = cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 0, lambda = 10^seq(10, -3, length = 100))
plot(cv.ridge)
```

![](Homework1_files/figure-gfm/ridge-1.png)<!-- -->

``` r
# best lambda
best.ridge = cv.ridge$lambda.min
best.ridge
```

    ## [1] 0.06892612

``` r
# test error
x.test = as.matrix(subset(test, select = -Solubility))
pred_ridge = predict(cv.ridge, s = best.ridge, newx = x.test)
mse_ridge = mse(test$Solubility, pred_ridge)
mse_ridge
```

    ## [1] 0.5121138

# Part c

``` r
# cross-validation
set.seed(2)
cv.lasso  = cv.glmnet(x.train, y.train, alpha = 1, lambda = 10^seq(10, -3, length = 100))
plot(cv.lasso)
```

![](Homework1_files/figure-gfm/lasso-1.png)<!-- -->

``` r
# best lambda
best.lasso = cv.lasso$lambda.min
best.lasso
```

    ## [1] 0.004534879

``` r
# test error
pred_lasso = predict(cv.lasso, s = best.lasso, newx = x.test)
mse_lasso = mse(test$Solubility, pred_lasso)
mse_lasso
```

    ## [1] 0.4998646

``` r
# coefficients of the final model
predict(cv.lasso, s = "lambda.min", type = "coefficients")
```

    ## 229 x 1 sparse Matrix of class "dgCMatrix"
    ##                              1
    ## (Intercept)        7.212941406
    ## FP001              .          
    ## FP002              0.237554622
    ## FP003             -0.047828666
    ## FP004             -0.234244319
    ## FP005              .          
    ## FP006             -0.074264043
    ## FP007              .          
    ## FP008              .          
    ## FP009              .          
    ## FP010              .          
    ## FP011              .          
    ## FP012             -0.054758453
    ## FP013             -0.051873443
    ## FP014              .          
    ## FP015             -0.094374257
    ## FP016             -0.072805709
    ## FP017             -0.141513899
    ## FP018             -0.094273168
    ## FP019              .          
    ## FP020              0.100791468
    ## FP021              .          
    ## FP022              .          
    ## FP023             -0.156238588
    ## FP024             -0.109785614
    ## FP025              .          
    ## FP026              0.264631778
    ## FP027              0.304514484
    ## FP028              .          
    ## FP029              .          
    ## FP030             -0.158914960
    ## FP031              0.120012007
    ## FP032              .          
    ## FP033              0.109219221
    ## FP034             -0.005463146
    ## FP035             -0.141191638
    ## FP036              .          
    ## FP037              0.208952547
    ## FP038              0.063363825
    ## FP039             -0.425644064
    ## FP040              0.434148993
    ## FP041              .          
    ## FP042              .          
    ## FP043              0.058040661
    ## FP044             -0.296631452
    ## FP045              0.089546154
    ## FP046              .          
    ## FP047              .          
    ## FP048              .          
    ## FP049              0.290168631
    ## FP050             -0.156457305
    ## FP051              .          
    ## FP052              .          
    ## FP053              0.237163059
    ## FP054             -0.084119169
    ## FP055             -0.142720343
    ## FP056              .          
    ## FP057             -0.088493182
    ## FP058              .          
    ## FP059             -0.294160494
    ## FP060              .          
    ## FP061             -0.160010778
    ## FP062              .          
    ## FP063              0.111700144
    ## FP064              0.240395105
    ## FP065             -0.139261493
    ## FP066              0.046618320
    ## FP067              .          
    ## FP068              0.001473962
    ## FP069              0.131756296
    ## FP070             -0.085768991
    ## FP071              0.095996776
    ## FP072              .          
    ## FP073             -0.121888929
    ## FP074              0.108060499
    ## FP075              0.185936681
    ## FP076              0.170871760
    ## FP077              0.081532084
    ## FP078             -0.145663270
    ## FP079              0.198457856
    ## FP080              .          
    ## FP081             -0.198774327
    ## FP082              0.139348941
    ## FP083             -0.343282524
    ## FP084              0.261617957
    ## FP085             -0.324029581
    ## FP086             -0.009397866
    ## FP087              .          
    ## FP088              0.098463046
    ## FP089              .          
    ## FP090              .          
    ## FP091              0.003514675
    ## FP092              .          
    ## FP093              0.153787051
    ## FP094             -0.166505319
    ## FP095              .          
    ## FP096             -0.048277607
    ## FP097              .          
    ## FP098             -0.050073264
    ## FP099              0.158421493
    ## FP100              .          
    ## FP101              .          
    ## FP102              0.001847309
    ## FP103             -0.114653291
    ## FP104             -0.082603903
    ## FP105             -0.059922120
    ## FP106              0.069717701
    ## FP107              .          
    ## FP108              .          
    ## FP109              0.323650105
    ## FP110              .          
    ## FP111             -0.349653584
    ## FP112             -0.004223788
    ## FP113              0.109649463
    ## FP114              .          
    ## FP115              .          
    ## FP116              0.024680548
    ## FP117              .          
    ## FP118             -0.097158662
    ## FP119              0.215510264
    ## FP120             -0.006456880
    ## FP121              .          
    ## FP122              0.205228161
    ## FP123              .          
    ## FP124              0.301724821
    ## FP125              0.050011585
    ## FP126             -0.154773851
    ## FP127             -0.498413559
    ## FP128             -0.225966983
    ## FP129              .          
    ## FP130             -0.277683306
    ## FP131              0.191428666
    ## FP132             -0.018189196
    ## FP133             -0.156263153
    ## FP134              .          
    ## FP135              0.193018319
    ## FP136              .          
    ## FP137              0.202550460
    ## FP138              0.232852291
    ## FP139              .          
    ## FP140              0.015344519
    ## FP141             -0.080184256
    ## FP142              0.451929504
    ## FP143              0.322004925
    ## FP144              .          
    ## FP145             -0.069901029
    ## FP146              .          
    ## FP147              0.145588786
    ## FP148             -0.049110323
    ## FP149              .          
    ## FP150              0.016984674
    ## FP151              .          
    ## FP152              .          
    ## FP153              .          
    ## FP154             -0.506999925
    ## FP155              0.026838008
    ## FP156             -0.228222519
    ## FP157             -0.065770355
    ## FP158              .          
    ## FP159              0.063345717
    ## FP160             -0.056630939
    ## FP161             -0.066583178
    ## FP162              .          
    ## FP163              0.174655584
    ## FP164              0.392312654
    ## FP165              .          
    ## FP166              0.021773490
    ## FP167             -0.096996203
    ## FP168              .          
    ## FP169             -0.147772339
    ## FP170              0.007732357
    ## FP171              0.246055654
    ## FP172             -0.541405327
    ## FP173              0.347341011
    ## FP174             -0.111905777
    ## FP175              .          
    ## FP176              0.403296103
    ## FP177              .          
    ## FP178              .          
    ## FP179              .          
    ## FP180             -0.084880307
    ## FP181              0.202247955
    ## FP182             -0.024850196
    ## FP183              .          
    ## FP184              0.309645305
    ## FP185              .          
    ## FP186             -0.207386888
    ## FP187              0.222389156
    ## FP188              0.209655065
    ## FP189              .          
    ## FP190              0.276789680
    ## FP191              0.082389646
    ## FP192              0.066451387
    ## FP193              .          
    ## FP194              .          
    ## FP195              .          
    ## FP196              .          
    ## FP197              .          
    ## FP198              0.160376098
    ## FP199              .          
    ## FP200              .          
    ## FP201             -0.292241423
    ## FP202              0.411830415
    ## FP203              0.074597990
    ## FP204              .          
    ## FP205              .          
    ## FP206             -0.054257119
    ## FP207              .          
    ## FP208              .          
    ## MolWeight         -1.327799081
    ## NumAtoms           .          
    ## NumNonHAtoms       .          
    ## NumBonds           .          
    ## NumNonHBonds      -0.981748364
    ## NumMultBonds      -0.135668498
    ## NumRotBonds       -0.237152803
    ## NumDblBonds        .          
    ## NumAromaticBonds  -0.097234922
    ## NumHydrogen        0.110046607
    ## NumCarbon         -0.638934035
    ## NumNitrogen        0.157565807
    ## NumOxygen          0.515047440
    ## NumSulfer         -0.256457665
    ## NumChlorine       -0.540229650
    ## NumHalogen         .          
    ## NumRings          -0.007673889
    ## HydrophilicFactor  .          
    ## SurfaceArea1       0.250426425
    ## SurfaceArea2       .

# Part d

``` r
library(pls)
```

    ## 
    ## Attaching package: 'pls'

    ## The following object is masked from 'package:caret':
    ## 
    ##     R2

    ## The following object is masked from 'package:stats':
    ## 
    ##     loadings

``` r
set.seed(3)
# fit PCR model using the function pcr()
fit.pcr = pcr(Solubility ~ .-Solubility, data = train, scale = TRUE, validation = "CV")
# summary(fit.pcr)
validationplot(fit.pcr, val.type = "MSEP", legendpos = "topright")
```

![](Homework1_files/figure-gfm/pc-1.png)<!-- -->

``` r
cv.mse  = RMSEP(fit.pcr)
ncomp.cv = which.min(cv.mse$val[1,,])-1
ncomp.cv
```

    ## 195 comps 
    ##       195

``` r
pred_pc = predict(fit.pcr, newdata = test, ncomp = ncomp.cv)
# test error
mse_pc = mse(test$Solubility, pred_pc)
mse_pc
```

    ## [1] 0.543476

# Part e

For ridge regression, the optimal lambda chosen is 0.0689261. For lasso,
the optimal lambda chosen is 0.0045349. For principal component
regression, the value of M chosen is 195. After using the test data to
calculate the mean square error for the four models (linear = 0.5558898,
ridge = 0.5121138, lasso = 0.4998646, principal component = 0.543476),
lasso has the smallest test error.

# Part f

I will choose lasso model for predicting solubility. First of all, lasso
yields the smallest mse on the test dataset. Moreover, the lasso
performs variable selection and yields sparse models. In this case, the
summary of coefficients for the final lasso model show that some of them
shrink to zero. Therefore, the lasso model also has higher
interpretability.
