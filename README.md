# JD 算法大赛

## 数据介绍
符号定义：
S：提供的商品全集；
P：候选的商品子集，P是S的子集；
U：用户集合；
A：用户对S的行为数据集合；
C：S的评价数据。

训练数据部分：
提供2016-02-01到2016-04-15日用户集合U中的用户，对商品集合S中部分商品的行为、评价、用户数据；提供部分候选商品的数据P。
选手从数据中自行组成特征和数据格式，自由组合训练测试数据比例。

预测数据部分：
2016-04-16到2016-04-20用户是否下单P中的商品，每个用户只会下单一个商品；抽取部分下单用户数据，A榜使用50%的测试数据来计算分数；B榜使用另外50%的数据计算分数。

为保护用户的隐私和数据安全，所有数据均已进行了采样和脱敏。
数据中部分列存在空值或NULL，请参赛者自行处理。

1. 用户数据

|user_id|用户ID|脱敏|
|--------------------------------|-----|------|
|age|年龄段|-1表示未知|
|sex| 性别 |0表示男，1表示女，2表示保密|
|user_lv_cd|用户等级 |有顺序的级别枚举，越高级别数字越大|
|user_reg_dt|用户注册日期|粒度到天|

2. 商品数据

|sku_id	| 商品编号	| 脱敏
|--------------------------------|-----|------|
|attr1	| 属性1	| 枚举，-1表示未知|
|attr2	| 属性2	| 枚举，-1表示未知|
|attr3	| 属性3	| 枚举，-1表示未知|
|cate	| 品类ID	| 脱敏|
|brand	| 品牌ID	| 脱敏|

3. 评价数据

|dt	| 截止到时间	| 粒度到天
|--------------------------------|-----|------|
|sku_id	| 商品编号	| 脱敏
|comment_num	| 累计评论数分段	| 0表示无评论，1表示有1条评论，2表示有2-10条评论，3表示有11-50条评论，4表示大于50条评论
|has_bad_comment	| 是否有差评	| 0表示无，1表示有
|bad_comment_rate	| 差评率	| 差评数占总评论数的比重

4. 行为数据

|user_id	| 用户编号	| 脱敏
|--------------------------------|-----|------|
|sku_id	| 商品编号	| 脱敏
|time	| 行为时间|
|model_id	| 点击模块编号，如果是点击	| 脱敏
|type	| 1.浏览（指浏览商品详情页）；2.加入购物车；3.购物车删除；4.下单；5.关注；6.点击
|cate	| 品类ID	| 脱敏
|brand	| 品牌ID	| 脱敏

任务描述：
参赛者需要使用京东多个品类下商品的历史销售数据，构建算法模型，预测用户在未来5天内，对某个目标品类下商品的购买意向。对于训练集中出现的每一个用户，参赛者的模型需要预测该用户在未来5天内是否购买目标品类下的商品以及所购买商品的SKU_ID。评测算法将针对参赛者提交的预测结果，计算加权得分。

## 作品要求
初赛提交CSV结果文件，进入复赛时提交源代码。
初赛提交CSV文件中包含对有购买意向的用户所购买商品的预测结果，字段如下：
user_id：用户ID，保证唯一，请勿在一次提交的结果文件中包含重复的user_id
sku_id：商品集合P中的商品ID，请勿在同一行中提交多个sku_id
对于预测出没有购买意向的用户，在提交的CSV文件中不要包含该用户的信息。

## Repo description
This repo contains several folders: data, result, plot.
data folder should contain all the raw data provided by JD.
result folder contain output csv files and plot folder contain output plots.
