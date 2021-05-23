from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
X = ("""
Claxton hunting first major medal

British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.

The 25-year-old has already smashed the British record over 60m hurdles twice this season, setting a new mark of 7.96 seconds to win the AAAs title.
"I am quite confident," said Claxton. "But I take each race as it comes. "As long as I keep up my training but not do too much I think there is a chance of a medal." Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage. Now, the Scotland-born athlete owns the equal fifth-fastest time in the world this year. And at last week's Birmingham Grand Prix, Claxton left European medal favourite Russian Irina Shevchenko trailing in sixth spot.

For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form.
In previous seasons, the 25-year-old also contested the long jump but since moving from Colchester to London she has re-focused her attentions. Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March.


""").lower() 
Y = ("""
Claxton hunting first major medal British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.
The 25-year-old has already smashed the British record over 60m hurdles twice this season, setting a new mark of 7.96 seconds to win the AAAs title.
"As long as I keep up my training but not do too much I think there is a chance of a medal."
For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form.
Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March.

""").lower() 
X_list = word_tokenize(X)  
Y_list = word_tokenize(Y) 
sw = stopwords.words('english')  
l1 =[];l2 =[]  
X_set = {w for w in X_list if not w in sw}  
Y_set = {w for w in Y_list if not w in sw} 
rvector = X_set.union(Y_set)  
for w in rvector: 
    if w in X_set: l1.append(1)
    else: l1.append(0) 
    if w in Y_set: l2.append(1) 
    else: l2.append(0) 
c6 = 0
for i in range(len(rvector)): 
        c6+= l1[i]*l2[i] 
cosine = c6 / float((sum(l1)*sum(l2))*0.5) 
print("similarity: ", cosine)


