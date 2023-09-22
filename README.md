# C0_topic_prediction 
The module contains the function (*textpreprocessed2topic*) that identifies the main (sub-)topics from Twitter posts in Italian about immigration.
The core of the component is a BERTopic model [[1]](#1) fine-tuned on a comprehensive set of tweets representing the information provided by both political entities and news sources on the immigration subject during the 5-year period 2018-2022 [[2]](#2). The model includes a pointer towards the model to be loaded in with SentenceTransformers, and returns as output the distribution(s) of the input text(s) over the 36 topics identified through the fine-tuning process. We further relied on *OpenAI* gpt-3.5-turbo to generate topic labels based on their best representing documents:

| Id | Label                                         | Id | Label                                 |
|----|-----------------------------------------------|----|---------------------------------------|
| 0  | Crime                                         | 18 | Sexual violence                       |
| 1  | Reception                                     | 19 | NGOs regulation                       |
| 2  | Sea rescue                                    | 20 | Racism and Hatred                     |
| 3  | Border crisis                                 | 21 | Illegal work and exploitation         |
| 4  | Landings                                      | 22 | Disease transmission                  |
| 5  | The Church's view                             | 23 | Cooperation and Development in Africa |
| 6  | Coronavirus and its spread                    | 24 | Right to asylum                       |
| 7  | First reception centers                       | 25 | EU regulation                         |
| 8  | Illegal immigration                           | 26 | Spread of fake news                   |
| 9  | Shanty towns                                  | 27 | Drugs and Drug Dealing                |
| 10 | Economy                                       | 28 | Movies                                |
| 11 | Humanitarian corridors for refugees           | 29 | LGBTQ+ and sexual minorities' rights  |
| 12 | Tourism and vacations                         | 30 | Quatargate                            |
| 13 | Pregnancy, parenthood, and children           | 31 | Luis Suarez's Italian exam scam       |
| 14 | Education and School-related themes           | 32 | Brexit                                |
| 15 | Vaccinations and EU digital Covid certificate | 33 | Sports                                |
| 16 | Islam                                         | 34 | Donations                             |
| 17 | Shipwrecks                                    | 35 | Marriages and Fake Unions             |

Access to the fine-tuned model is provided through the *HuggingFace* platform, together with the list of training hyperparameters and framework versions [https://huggingface.co/brema76/topic_immigration_it].

The module takes as input the output dictionary of the C0_text_preprocessing component. Namely, it first receives all the preprocessed versions of an input tweet whose topic has to be detected. Then it selects only the value of the item corresponding to the key *topic*. The output consists of: a list of 0 ≤ *p* ≤ 36 *labels*, each describing a topic; a list [*item*<sub>*t*</sub>] ∈ *R*<sup>*p*</sup> whose entries, one per topic, represent the probability that the corresponding topic is present in the input text.
The fine-tuned model can be loaded directly from the HuggingFace hub: 

    from bertopic import BERTopic
    topic_model = BERTopic.load('brema76/topic_immigration_it')

The module contains a Flask app that calculates the topic distribution for the text corresponding to the value of the item identified by the key *topic* in the output dictionary of the C0_text_preprocessing component. In addition, it receives a parameter *θ* (default set to 0) that limits the output to those topics whose score probability is larger than *θ*.

A working example launching a Flask server to listen on port *127.0.0.1:5000* is reported in the *example.py* file.

## References
<a id="1">[1]</a> 
Grootendorst M (2022). 
BERTopic: Neural topic modeling with a class-based TF-IDF procedure. 
arXiv preprint arXiv:2203.05794

<a id="2">[2]</a> 
Brugnoli E, Gravino P, Prevedello G (2023).
Moral Foundations Theory's values in social media post.
VALE workshop at ECAI (accepted)

