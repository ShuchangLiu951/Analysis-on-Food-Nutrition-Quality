# A way to predict nutrition level based on recipe

by Kevin Li (kel062@ucsd.edu) & Shuchang Liu (shl153@ucsd.edu)


---

## Introduction

In this project, we explore two datasets that consist of recipes and user interactions. The datasets provides detailed information about recipes, including their ingredients, preparation steps, nutritional values, and tags, etc., as well as user reviews and ratings. This enables us to investigate cooking trends, recipe complexity, and user preferences.
The recipes dataset contains key information about recipe ingredients, tags, cooking steps, and nutrition that has 83,782 rows, while the interaction dataset includes user reviews and ratings that has 731,927 rows. The central question we are going to explore is: “Can we predict whether a recipe will be rated as healthy, normal, or bad based on its nutrition?” This question is important for promoting health-conscious choices, improving recipe recommendations, making it relevant to users alike.
The following columns are critical for our analysis as they provide information about recipe characteristics and nutritional properties:
id: A unique identifier for each recipe.
minutes: The total time required to prepare the recipe, in minutes.
tags: tags that describe the recipe (e.g., "30-minutes-or-less", "vegetarian").
n_steps: The number of steps required to prepare the recipe.
ingredients: A list of all the ingredients used in the recipe.
n_ingredients: The number of ingredients in the recipe.

---

## Cleaning and EDA

In this step, we carefully cleaned the data to ensure it was ready for analysis. Missing values in the rating column were replaced with NaN to prevent inaccuracies. A new column, nutrition_score, was created to classify recipes as "healthy," "normal," or "bad" based on the proportions of beneficial and harmful nutrients relative to calorie content. This step also involved splitting and cleaning string-based columns such as tags and ingredients to enhance their usability. We then merged the recipes and interaction datasets on the recipe ID to allow for combined analysis.

### result

| Column     |   Description |
|:------------|--------:|
|   id | <class 'numpy.int64'>   |
|   rating average | <class 'numpy.float64'>   |
|   name | <class 'str'>   |
|   minutes | <class 'numpy.int64'>   |
|   contributor_id | <class 'numpy.int64'>   |
|   submitted | <class 'str'>   |
|   tags | <class 'str'>   |
|   nutrition | <class 'list'>   |
|   n_steps | <class 'numpy.int64'>   |
|   steps | <class 'str'>   |
|   description | <class 'str'>   |
|   ingredients | <class 'str'>   |
|   n_ingredients | <class 'numpy.int64'>   |
|   user_id | <class 'numpy.float64'>   |
|   recipe_id | <class 'numpy.float64'>   |
|   date | <class 'str'>   |
|   rating | <class 'numpy.float64'>   |
|   review | <class 'str'>   |
|   nutrition_score | <class 'numpy.float64'>   |
|   missing_rating | <class 'numpy.bool'>   |

### Univariate Analysis
<iframe src="assets/number_step.html" width=800 height=400 frameBorder=0></iframe>

We explored the distribution of individual columns to understand recipe characteristics. A histogram of the n_steps column revealed that most recipes require fewer than 20 steps, with a sharp decline in recipes needing more steps. This suggests that simpler recipes dominate the dataset, which might reflect user preferences or the platform's content trends.

### Bivariate Analysis
<iframe src="assets/file-name.html" width=800 height=400 frameBorder=0></iframe>

We examined relationships between pairs of variables to uncover potential associations. For instance, we compared the average protein content of "main-dish" recipes to "appetizer" recipes by grouping data based on tags and calculating the mean protein percentage of the daily value. A bar plot showed that main dishes typically have higher protein content than appetizers.

### Aggregates

| nutrition_score     |   rating |
|:------------|--------:|
|   'bad' | 4.68   |
|  'normal' | 4.69  |
|   'healthy' | 4.64   |

To identify notable trends, we grouped data by the nutrition_score column and calculated the mean user rating for each group. Surprisingly, recipes classified as "normal" received slightly higher average ratings (4.69) than "healthy" (4.64) or "bad" (4.68) recipes. This finding suggests that user ratings are not solely dependent on a recipe's healthiness.



---

## Assessment of Missingness

### Missing Dependency

>Number of ingredients and Rating

**Null Hypothesis:** The missingness of ratings does not depend on the number of ingredients in the recipe.

**Alternative Hypothesis:**  The missingness of ratings does depend on the number of ingredients in the recipe.

**Test Statistic:** The absolute difference of mean in the proportion of number of ingredients of the distribution of the group without missing ratings and the distribution of the group with missing ratings.

**Significant Level:** 0.05

We ran the permutation test by shuffling the missingness of rating 500 times to collect 500 simulating mean differences in the two distributions as described in the test statistic.
<iframe src="assets/missing-on-n_ingredients.html" width=800 height=400 frameBorder=0></iframe>

The observed test statistic 0.0 is indicated at the end of graph. p-value(0.0) < 0.0. **We reject the null hypothesisthe**. The missing rating does depend on the number of ingredients in the recipe.


>Number of steps and Rating

**Null Hypothesis:** The missingness of ratings does not depend on the number of steps in the recipe.

**Alternative Hypothesis:**  The missingness of ratings does depend on the number of steps in the recipe.

**Test Statistic:** The absolute difference of mean in the proportion of number of steps of the distribution of the group without missing ratings and the distribution of the group with missing ratings.

**Significant Level:** 0.05

We ran the permutation test by shuffling the missingness of rating 500 times to collect 500 simulating mean differences in the two distributions as described in the test statistic.

<iframe src="assets/missing-on-minutes.html" width=800 height=400 frameBorder=0></iframe>

The observed test statistic 0.0 is indicated at the end of graph. p-value(0.128) > 0.0. **We fail reject the null hypothesisthe**. The missing rating does depend on the number of ingredients in the recipe.

---

## Hypothesis Testing

We conducted a hypothesis test to evaluate whether the mean protein content significantly differs between recipes categorized as "main-dish" and "appetizers." The null hypothesis posits that the difference in mean protein content between the two categories is due to random chance, while the alternative hypothesis states that the mean protein content for "main-dish" recipes is significantly higher than that of "appetizers." Using a permutation test with 10,000 iterations, we calculated the observed difference in means and compared it to the distribution of differences generated by shuffling category labels.
The results showed an observed difference of approximately 35.8, with a p-value of 0, indicating statistical significance. The histogram of permutation differences confirms that the observed difference lies far outside the null distribution. These results lead us to reject the null hypothesis, concluding that "main-dish" recipes have significantly higher protein content compared to "appetizers." This finding aligns with expectations, as main dishes are generally more protein-rich.

---

## Prediction Problem

Our prediction problem involves determining whether a recipe is healthy, normal, or bad based on its characteristics such as ingredients, tags, and nutritional information. This is a multiclass classification problem, as the response variable (nutrition_score) has three distinct categories. We chose this response variable because it quantifies the overall nutritional quality of recipes, making it central to assessing their healthiness. The evaluation metric for the model is accuracy, as it provides a straightforward measure of the proportion of correctly classified samples. While metrics like F1-score are valuable for imbalanced datasets, the classes in our dataset are relatively balanced, making accuracy a suitable and interpretable choice. This framing ensures that the model effectively distinguishes between the three health categories, supporting healthier dietary decisions.

___

## Baseline Model

We implemented a baseline model to predict the nutrition_score of recipes (healthy, normal, or bad) using a multiclass classification approach. The model included three features: two nominal features (ingredients and tags) and two quantitative features (n_steps and n_ingredients). For preprocessing, we used a ColumnTransformer to apply CountVectorizer on the nominal features, converting textual data into numerical representations, and Binarizer on the quantitative features to handle threshold-based transformations. A DecisionTreeClassifier (with a maximum depth of 15) was used as the estimator, all encapsulated within an sklearn pipeline to streamline feature transformation and model training. The model achieved a training accuracy of 79.16% and a test accuracy of 71.80%, demonstrating moderate generalizability.



___

## Final Model

Final Model
To improve upon the baseline model, we added new features and optimized hyperparameters using GridSearchCV. The final model incorporated the same nominal (ingredients and tags) and quantitative features (n_steps and n_ingredients) as the baseline but introduced feature scaling for numerical attributes and hyperparameter tuning for the decision tree classifier. We used a grid search to identify the best combination of hyperparameters, optimizing for the split criterion (gini or entropy), the maximum tree depth, and the minimum samples per leaf. The optimal parameters were: criterion='entropy', max_depth=10, and min_samples_leaf=8. The final model achieved a cross-validation accuracy of 70.91% and a test accuracy of 71.57%, comparable to the baseline but slightly more refined due to better generalization from hyperparameter optimization.

___

## Fairness Analysis






