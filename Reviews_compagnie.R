####SENTIMENT ANALYSIS OF 


#LOAD NECESSARY LIBRARIES
library(RedditExtractoR)
library(dplyr)
library(ggplot2)
library(tidyr)
library(cld2)
library(cld3)
library(pushshiftR)
library(tm)
library(udpipe)
library(rtweet)
library(ngram)

#Load some useful functions
source("utility.R") 

#Load functions for sentiment analysis
source("sentimentFunctions.R") 


## Load a pre-trained model for dictionaries and the lemmatizzation in the italian language (NOTE: this model wa trained by the UD treebank for italian)
un_model_I <- udpipe_load_model("italian-vit-ud-2.5-191206.udpipe") 

#Common compound
load("frmComposteOpSy.RData") 

#Common stopwords
load("IT_stopwwords.RData") 

#Dictionaries for sentiment
load("diz_polarity.RData") 

# 1. LOADING DATA, TEXTUAL CLEANING AND CORRECTION OF COMPOUND EXPRESSIONS
#Load data of the reviews
load("dfRecHoEoFw.RData")

#Clean text
dfRecHoEoFw$txtp <- cleanText(xtxt = dfRecHoEoFw$testo, punctation = T) 

#Visualize N-Grams
visNGram(x = dfRecHoEoFw$txtp ,ngrI = 2,ngrF = 4,nn = 50)

#Create a correction vector to store some compound expression
vcorrez <- c("servizio clienti", NA,
             "il call center", "call_center",
             "call center", NA,
             "velocità di navigazione", "velocità_navigazione",
             "il servizio clienti", "servizio_clienti",
             "problemi di connessione", "problemi_connessione",
             "rapporto qualit  prezzo", NA)      

#Correct some compund expression to make the process easier 
dfRecHoEoFw$txtp <- corFrmComp(dfRecHoEoFw$txtp, correzioni = vcorrez) 

visNGram(dfRecHoEoFw$txtp, 2, 4)
dfRecHoEoFw$txtp <- corNGram(dfRecHoEoFw$txtp, verbose=T) # correzione frm cmpst

# 2. Lemmatization and deleting stopwords 
dfRecHoEoFw$doc_id <- 1:nrow(dfRecHoEoFw)
dfRecHoEoFwL <- lemmaUDP(x = dfRecHoEoFw$txtp,
                         model = un_model_I,
                         doc_id = dfRecHoEoFw$doc_id,
                         stopw = stopwIT,
                         userstopw = c("homobile", "eolo", "fastweb"))
View(dfRecHoEoFwL)

# 3. Reconstruction of lemmatized sentences 
txtL <- dfRecHoEoFwL%>%
  mutate(doc_id=as.numeric(doc_id)) %>%
  filter(!is.na(lemma) & STOP==FALSE & upos %in% c("ADJ","NOUN","PROPN","VERB","ADV")) %>%
  group_by(doc_id) %>%
  summarise(txtL=paste(lemma,collapse = " "))

dfRecHoEoFw <- left_join(dfRecHoEoFw, txtL, by="doc_id")
View(dfRecHoEoFw)

# 4. Reviews that are significant (let's say with at least 5 lemmas)
#Count lemmas per senteces
nlemmi <- sapply(dfRecHoEoFw$txtL, FUN=wordcount)  

#Select sentences with at least 5 lemmas 
dfRecHoEoFw_ <- dfRecHoEoFw[nlemmi>4,]   
View(dfRecHoEoFw_)

# 5. 10 most common lemmas and worldcloud
dfRecHoEoFw_$txtL <- removeWords(tolower(dfRecHoEoFw_$txtL), words=stopwITwc)

corpL_ <- Corpus(VectorSource(dfRecHoEoFw_$txtL))
tdmL_ <- TermDocumentMatrix(corpL_)
tdmL_M <- as.matrix(tdmL_)     # MATRIX terms-documents

# distribution of frequences of lemmas
dfcorpL_ <- data.frame(words=rownames(tdmL_M),        
                       freq=rowSums(tdmL_M)) %>%      
  arrange(-freq)
rownames(dfcorpL_) <- NULL
head(dfcorpL_, 10)

library(wordcloud)

# WORDCLOUD LEMMAS weighted TF (Term Frequency) --> frequence of a word in a single document
par(mar=c(0,0,0,0))
wordcloud(words = dfcorpL_$words,
          freq = dfcorpL_$freq,             
          max.words = 80,                  
          random.order = FALSE,              
          colors = brewer.pal(8, "Accent"))
text(0.5,1,"Wordcloud lemmas - TF",cex=1.,font = 2)


# WORDCLOUD LEMMI PONDERATA TF-IDF --> We add the frequence of the word inside the total corpus of documents
tdmL_IDF <- weightTfIdf(tdmL_)

tdmL_IDFM <- as.matrix(tdmL_IDF)
dfcorpL_IDF <- data.frame(words=rownames(tdmL_IDFM),
                          tdmL_IDFM,
                          freq=rowSums(tdmL_IDFM)) %>%
  arrange(-freq)
rownames(dfcorpL_IDF) <- NULL

par(mar=c(0,0,0,0))
wordcloud(words = dfcorpL_IDF$words,
          freq = dfcorpL_IDF$freq,
          max.words = 80,
          scale = c(2.5,0.3),
          random.order = F,
          colors = brewer.pal(n = 6,name = "Accent"))
text(0.5,1,"Wordcloud lemmas weighted - TFIDF",cex=1.5,font=2)








# 6. COMPARISON CLOUD of the 3 companies with TD-IDF 
dfRecHoEoFw.company <- dfRecHoEoFw %>% 
  group_by(company) %>% 
  summarise(txt=paste(txtL,collapse = " "))
corpL.company <- Corpus(VectorSource(dfRecHoEoFw.company$txt))
tdmL.company <- TermDocumentMatrix(corpL.company)

tdmLIDF.company <- weightTfIdf(tdmL.company)
tdmL.companyM <- as.matrix(tdmL.company)
colnames(tdmL.companyM) <- dfRecHoEoFw.company$company

par(mar=c(0,0,0,0))
comparison.cloud(term.matrix = tdmL.companyM,
                 scale = c(2,0.2),
                 max.words = 80,
                 colors = c("Blue","Black","Green"),
                 match.colors = T,
                 title.size = 1)
text(0.5,1,"Comparison cloud TFIDF of the 3 companies",font=2)


# 7 Computing the Average Sentiment Score using sentiment dictionaries: SYUZHET, OPENR, NCR
#Defining the list of dictionaries we need to use
lDizSent <- list(Syuzhet = dSyuzB, OpenR = dOpenR, NCR = dNcr)    

dfRecHoEoFwLPOL <- sentiMediaDiz(x = dfRecHoEoFwL,           # output dof the lemmatization
                                 dict = lDizSent,              # dict to use
                                 negators = polarityShifter,
                                 amplifiers = intensifier,
                                 deamplifiers = weakener)

dfRecHoEoFwLPOL$DizioPol
dfRecHoEoFwLPOL$MediaPol

#Let's add the Average Score of Sentiment to the dataframe
dfRecHoEoFw_Sent <- left_join(dfRecHoEoFw_,                   
                              dfRecHoEoFwLPOL$MediaPol,         
                              by="doc_id")  

#Computing class of polarity
dfRecHoEoFw_Sent$cl_mediaPol <- ifelse(dfRecHoEoFw_Sent$mediaSent<0,             
                                       "Negative",                                      
                                       ifelse(dfRecHoEoFw_Sent$mediaSent>0,
                                              "Positive",
                                              "Neutral"))

# 8. Average scores and graphic with distribution of the classes of polarity
#Total mean of sentiment score
mean(dfRecHoEoFw_Sent$mediaSent) 

#Average sentiment score grouped by company
dfRecHoEoFw_Sent %>% group_by(company) %>%          
  summarise(mean(mediaSent))                      


#Distribution of the class of polarity divided per period?
dfRecHoEoFw_Sent %>% group_by(company,cl_mediaPol) %>%            
  summarise(n=n()) %>%                                            
  mutate(perc=n/sum(n)*100) %>%                                          
  ggplot(aes(x=company,
             y=perc,
             fill=cl_mediaPol))+
  geom_col()+
  theme_light()+                      
  ggtitle("Distribution of the polarity classes per company")


# 9. COMPARISON CLOUD USING ONLY NEGATIVE AND POSITIVE
dfRecHoEoFw_polar <- dfRecHoEoFw_Sent %>% 
  group_by(cl_mediaPol) %>% 
  summarise(txt=paste(txtL,collapse = " "))
corpL.polar <- Corpus(VectorSource(dfRecHoEoFw_polar$txt))
tdmL.polar <- TermDocumentMatrix(corpL.polar)
tdmL.polarM <- as.matrix(tdmL.polar)
colnames(tdmL.polarM) <- dfRecHoEoFw_polar$cl_mediaPol

par(mar=c(0,0,0,0))
comparison.cloud(term.matrix = tdmL.polarM[,c(1,3)],
                 scale = c(2,0.2),
                 max.words = 80,
                 colors = c("Black","green"),
                 match.colors = T,
                 title.size = 1)
text(0.5,1,"Comparison cloud TF of the two polarity classes",font=2)






# 10. COMPUTING GRADE CLASSES AND COMPARING CPOLARITY CLASSES
#Defining grade classes
dfRecHoEoFw_Sent$cl_voto <- cut(dfRecHoEoFw_Sent$voto,          
                                breaks = c(1,2,3,5),              
                                include.lowest = T)

#table to compare grade classes and polarity
tabvp <- table(dfRecHoEoFw_Sent$cl_voto, dfRecHoEoFw_Sent$cl_mediaPol)   
addmargins(tabvp)                                                     
sum(diag(tabvp))/sum(tabvp)    #accuracy

# 11. EMOTIONS DISTRIBUTIONS
dfRecHoEoFw_emo <- myClassEmotion(textColumns = dfRecHoEoFw_$txtL,
                                  algorithm = "bayes",
                                  lexicon = "emotions_it_lem.csv")
dfRecHoEoFw_emo$documenti
dfRecHoEoFw_emo$documenti %>% group_by(best_fit) %>%
  summarise(n=n()) %>%
  mutate(perc=n/sum(n)*100)


dfRecHoEoFw_emo$documenti %>% group_by(best_fit) %>% 
  summarise(n=n()) %>% 
  mutate(perc=n/sum(n)*100) %>% 
  ggplot(aes(x=reorder(best_fit,perc),
             y=perc,
             fill=best_fit))+
  geom_col()+
  theme_light()+
  scale_fill_brewer(palette = "Set1")+
  coord_flip()+
  labs(title = "sentiment emotions",
       subtitle = "emotions lexicon - naiveBayes")+
  theme(legend.position = "none")+
  xlab(NULL)+
  ylab("%")
