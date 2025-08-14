*Fast Fashion vs. Sustainable Consumption:*
*A Battle Between Long Lasting Garments, Environmental Impact, and Customer Sentiment*

**Introduction:**

92,000,000 tonnes, that’s over 36 million elephants, 1,700 titanics, and 250 Empire State Buildings. That is how much textile (clothing) waste is produced each year. Needless to say, that is an alarming amount. This number has only grown over the years, with an 811% increase since 1960. This is due in part to the rise in popularity of “fast fashion”, clothing companies which have incredibly high output, but notoriously low prices and low quality of goods, leading a consumer to be more likely to throw away an item. You’ve probably been scrolling on TikTok or Instagram reels and have seen a video promoting a new clothing product, and a sponsored link in the description. More often than not, those clothing products are just one of the thousands of trending products that a fast fashion company has produced in a given month. These are called microtrends in the fashion community, niche things that are popular for a few weeks to a month, then fade into your closet or the trash can, never to be used again. I’ve fallen victim to this, just as many others have, and it proved to me that this issue has propagated deep into the lives of consumers. UNEP (UN Environment Program) researchers found in their research that, “the number of times a garment is worn has declined by around 36% in 15 years,” Truly showing the impact of these trends. This began to form a question in my head, “why has fast fashion truly contributed to textile waste?”

**The Main Questions to Answer:**

I began with refining my question; “why has fast fashion truly contributed to textile waste” is not a good question to answer with data. I found that tackling such a large problem will require splitting my question into sections. But what sections would help answer?

First, I looked at the intersection; what do fast fashion and sustainable labels have in common? They both appeal to wider demographics, both promote their cost per wear, and actively market through social media. But these do not reveal the crux of the matter. Though, they give us a hint of where to look.

The divisiveness comes from the difference between the two. Sustainable brands market high quality materials, long lasting products, and environmentally sustainable practices. While on the other hand, fast fashion companies have a high turnover rate following microtrends, low durability, and poor materials. 

This apparent contradiction, where fast fashion thrives despite its clear drawbacks, leads to the core questions this project answers: 
How does consumer sentiment in reviews correlate with objective quality metrics?

What patterns emerge when price and quality are analyzed together?

What extent is greenwashing (marketing an unsustainable product as sustainable) a factor in this dynamic?

**Data & Methodology:**

The first step I took was to loosely define a schema for my data; I needed to figure out exactly what data I would need. This was fairly intuitive, as my research questions already defined the necessary components. For products, I needed to find: brand, price, description, and composition, which I would use Pandas and scikit-learn to operate on. 

I combined multiple heterogeneous datasets that required significant cleaning and integration. I gathered 6 main product datasets, through existing resources and custom scrapers. I found ~180mb of Kaggle data sets with varying schemas for SHEIN, Zara, H&M, and Patagonia product details, then created custom web scrapers for Everlane and Reformation. This resulted in 238mb of uncleaned, differing schema, and poorly formatted data. 

To create workable data, I created a normalized Pandas dataframe with 7 columns: brand, price, materials, description, title, reviews, and id. To format, I used a dictionary based on the similarly named columns in my raw data. For example, [“brand”, “Brand”, “seller”, and “store”] are all mapped to the “brand” column in the normalized set. To clean, I would infer null data from the .csv file it came from, only when the inference exists. Then, I normalized the price using a regular expression to strip non-digits then used the pd.to_numeric to shift everything to float. Similarly, I normalized materials, splitting on nondigits then renormalizing to [0, 1] (0.3 cotton, 0.7 polyester). This gave me an extremely strong starting point to begin data analysis, but first, I needed to source my sentiment data to cross-examine the points.

For brand and product sentiment, Reddit was my best source to find customer opinions. I found them through the subreddits r/<mens/womens>fashion and r/<male/female>fashionadvice. I chose to use a Spark data pipeline and VADER sentiment analysis on the Reddit data, giving me highly reliable and normalized data.

To process the large volume of Reddit sentiment data, I engineered a pipeline on SFU’s compute cluster using Apache Spark. This allowed for efficient, parallel ingestion and processing of submissions and comments by partitioning the data by data and subreddit.

Now that I had clean product data, I began researching the observed durability of products using review data. This led me to create a weighted durability model using a bag-of-words proxy, which groups together “bags” of words into positive and negatively correlated groups. For example, a positively correlated word would be “sturdy”, or “durable”, while “fell apart”, or “ripped” would be negatively correlated. Then, for each product, I assigned a durability score, which was the sum of positive words minus the sum of negative words, then normalized over total reviews (D = 1num(t)(tPcount(t) -tNcount(t)) where D = observed durability score, t = comment, P = positive words, N = negative words. 

	Next, I needed to analyze objective material quality, and give a numerical value for how high quality a product was. I began by giving scores to individual materials, with higher quality materials getting higher scores, like lyocell/hemp being 9.0, and virgin polyester being a 2.0. This then allowed me to create another dictionary which would handle synonyms and canonicalize terms. I mapped things like elastane to spandex, organic cotton to cotton, and distinguished things like recycled polyester vs. virgin polyester. To weight material quality, I summed over the scores of (weighted) listed materials of the product, 
Q =i=1kfᵢ * sᵢ , where Q = objective material quality, k = total listed materials, f = normalized fraction of material, s = the mapped score per material. 

	These data points now allowed me to cross examine the effect they have on sentiment. Using the same review corpus, I used a polarity-aware model (VADER) to score each review on the interval [-1, 1] and aggregated it to the product level by taking the mean compound score. For a product p, reviews Rₚ, the sentiment score is Sₚ=1|Rₚ| rRₚVADER_compound(r). This yields a length and scale-stable sentiment score that 
is comparable across all products.

Results & Findings:

	With both durability and quality metrics in hand, I quantified their association with observed sentiment. The analysis revealed a weak but positive Pearson correlation (r = 0.1563) between objective material quality and sentiment and a p-value extremely close to zero. Then, I created a scatterplot + regression fit in matplotlib and seaborn (as seen on right) to observe linearity and spread across various levels of material quality. While a positive trend exists, quality alone is not a strong predictor of satisfaction, as the p-value shows a significant chance of randomness.


Next, I found the Pearson correlation between sustainability claims and actual material quality. The data shows a near-zero correlation, with nearly identical violin plots (as seen on right). This indicates that “green” marketing claims are not backed by superior materials, showing clear evidence of greenwashing in product marketing. Any product could be called “green” and have the same materials as one that has no sustainable marketing at all.

Using the scatterplot and regression line, I was able to create feature vectors comprising material quality, durability, and review sentiment. Standardizing features to a mean of zero and unit variance, I was able to apply K-means clustering. I summarized each cluster by its mean along the original feature scale, giving way to four main clusters, showing that the market is not a simple binary, but is composed of different segments. These segments are: 1. The Reliable Performers: products with medium quality, high durability and sentiment, and medium price. 2. The Truly Sustainables: high material quality, durability, and price, but medium sentiment. 3. The Disappointments: medium quality and price, high durability, low sentiment. And finally, 4. The Delicate Luxuries: high price and quality, but low durability, with extremely varied sentiment. These unintuitive patterns begin to emerge when observing the data based on its market segments. There is much more that goes on besides a “good” and “bad” product. For example, a reliable performer might be a requirement for a job, and durability is its main function.

Limitations:


	Though my analysis and data was enough to draw significant conclusions, the fashion market is so complex that some data points were simply out of my scope to obtain and process within a semester. For example, my durability scores and material quality scores were proxies, not direct real-world measurements. Much of the data was gathered from subjective sources, and thus no piece of data was truly normalized. On top of this subjectiveness, there was slight data bias present in my sentiment data. Users of fashion subreddits are more likely to be knowledgeable and strongly opinionated than the average consumer, so claims of poor quality may be exaggerated.

Given more time to do this project, I would look more towards the financial side of this. If I had access to the income statements, cash flow statements, and operating statements of each company, I would be able to paint a clearer picture for things like advertising and market share. I would also be able to include more big hitter companies. I only covered a specific selection of large brands, which leaves the possibility that the global fashion market has different statistics than the ones I measured. 

Lastly, companies like SHEIN are notorious for giving customers discounts for leaving positive reviews on their products, creating an intense feedback loop where customers are encouraged to buy and review, making others believe it is because the products are good. Given time, I could’ve created an NLP tool that is able to discern real positive reviews to fake ones.

Conclusion
	
	Based on the evidence gathered by the secondary questions, the relationship between price, quality, satisfaction, and environmental impact is complex and counterintuitive. Consumer perceptions frequently misalign with objective reality. The price-quality relationship does not have a simple direct correlation with consumer satisfaction. The highest levels of satisfaction were not found in the highest-priced or highest-quality items, but in a sweet spot of value, where mid-range products exceeded expectations on durability. This shows that satisfaction is a function of expectation and performance, not just quality. 

Furthermore, consumer perceptions are poorly aligned with sustainability metrics, demonstrating that marketing claims have no correlation on true quality of garments. This reveals a significant disconnect between marketing narratives and product substance, where companies can leverage the imagery of sustainability to drive consumption without being held accountable for the reality of their products.

This project reveals a marketplace where consumer trust is the target, marketing is the weapon, and the actual quality and lifetime of the clothing is often the first casualty. 

Work Cited
Putting the brakes on fast fashion., UNEP, https://www.unep.org/news-and-stories/story/putting-brakes-fast-fashion 