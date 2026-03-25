Attempt feedback

G049: 

The report is well-motivated and already has a strong engineering direction. 

The problem is explained clearly (even for a reader encountering the project for the first time, given that you haven’t made it to any of the tutorials). 

You have identified the relevant failure modes and built a pipeline from a real API, which is impressive to see. 

For the final report, make sure you document your evaluation metrics and strategies in a precise, reproducible way. 

The main gaps to address before the final version are: 
to explain concretely how gaze will be incorporated into training and how you will handle view-related variation (e.g., normalising across camera angles or sticking to a consistent view), and (2) pin down the transformer fusion design (architecture choice, what constitutes a “sequence”, how gaze enters the model, and the final output label space). 

Also, report dataset scale and distribution more explicitly: number of matches, legs, throws, players, competitions, and timeframe, and discuss class imbalance; otherwise, accuracy can be misleading if a few targets dominate. 

For the gaze dataset, be explicit about sample size and how representative it is, so it’s clear whether the gaze is only descriptive or supports predictive modelling. 

Another area that needs attention is citations: there are currently very few. It’s unclear whether this is because the work is highly novel or because a fuller literature review is still pending. 

For the final report, you should expand related work, especially on gaze estimation, intent modelling, sports broadcasting automation, and multimodal fusion. 

Finally, because the topic is niche and terminology can get confusing, it would help to add an appendix that includes the exact feature list, key statistics, and a glossary to standardise terms like “segment,” “wedge,” “number outcome,” and “region.”