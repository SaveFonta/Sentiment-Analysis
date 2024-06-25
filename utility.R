# -----------------------------------------------------------------------------------------------
# funzioni presenti nello script:
#
# search_reddit      ricerca reddit usando RedditExtractoR
# search_reddit2     ricerca reddit usando RedditExtractoR, con selezione da subreddit senza keywords e ricodifica UTF-8 del campo txt
# reddit.users.info  ritorna informazioni su utenti reddit
# data2timestamp     converte data in timestamp
# timestamp2data     converte timestamp in data
# det.lang.2         determina se un testo è nella lingua selezionata
# search.Pushshift   ricerca reddit usando pushshiftR
# cleanText          pulizia di un testo
# corNGram           modifica forme composte sulla base di dizionario (solo italiano)
# visNGram           visualizza n-grammi per individuare forme composte
# corMultWord        correzione delle forme composte da file excel
# corFrmComp         modifica delle forme composte da vettore di correzioni
# multWordPOS        trova forme composte in base ad analisi grammaticale
# lemmaUDP           lemmatizzazione
# make.tweet.df      crea un data.frame dal risultato di una funzione search_tweets o get_timeline di rtweet
# my_get_timeline    get_timeline per più utenti twitter
# -----------------------------------------------------------------------------------------------

vPackg <- c("rtweet","ngram","dplyr","openxlsx","udpipe","RedditExtractoR","pushshiftR","stringr")
if(sum(vPackg %in% as.data.frame(installed.packages(.libPaths()))[,1]) != length(vPackg)){
  manc=vPackg[vPackg %in% as.data.frame(installed.packages(.libPaths()))[,1]==F]
  warning(paste("deve essere installata la libreria:",manc))
}


cleanText <- function(xtxt = NULL, 
                      hashtag = TRUE, 
                      mention = TRUE, 
                      numbers = FALSE, 
                      punctation = FALSE,
                      lowercase = TRUE){
  # function parameter description
  # xtxt = vector of texts (tweets)
  # hashtag = logical, if TRUE removes the entire hashtags
  # mention = logical, if TRUE removes all mentions @
  # numbers = logical, if TRUE removes all numbers from messages
  # punctation = logical, if TRUE removes punctation
  # lowercase = logical, if TRUE changes all texts to lowercase
  #
  # check if x is defined and is an object of type character
  if(is.null(xtxt) | class(xtxt) != "character"){stop("invalid character vector")}
  # check if the other arguments are logical
  if(!is.logical(hashtag) | !is.logical(hashtag) | !is.logical(numbers) | !is.logical(punctation) | !is.logical(lowercase)){
    stop("invalid argument")}
  # html symbols vector
  htmlsymbols <- c("&copy;","&reg;","&trade;","&ldquo;","&lsquo;","&rsquo;","&bull;",
                   "&middot;","&sdot;","&ndash;","&mdash;","&cent;","&pound;","&euro;",
                   "&ne;","&frac12;","&frac14;","&frac34;","&deg;","&larr;","&rarr;",
                   "&hellip;","&nbsp;","&lt;","&gt;","&amp;","&quot;")
  htmlsymbolsU <- paste(htmlsymbols,collapse = "|")
  
  # remove links
  xtxt = gsub("(f|ht)(tp)(s?)(://)(.\\S+)[.|/](.\\S+)", " ", xtxt)
  # remove references in retweets
  xtxt = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", " ", xtxt)
  xtxt = gsub("(rt|via)((?:\\b\\W*@\\w+)+)", " ", xtxt)
  # html symbols
  xtxt = gsub(htmlsymbolsU," ",xtxt)
  # punctation
  if(punctation == TRUE){
    xtxt = gsub("([#@])|[[:punct:]]", " \\1", xtxt)
  }
  # control characters
  xtxt = gsub('[[:cntrl:]]', ' ', xtxt)
  # those that are not graphic characters (what is not [[:alnum:][:punct:]])
  xtxt = gsub('[^[:graph:]]', ' ', xtxt)
  # hashtag
  if(hashtag == TRUE) xtxt = gsub("#\\S+", " ", xtxt)
  # mention
  if(mention == TRUE) xtxt = gsub("@\\S+", " ", xtxt)
  # numbers
  if(numbers == TRUE) xtxt = gsub("[[:digit:]]", "", xtxt)
  # tabulations and more spaces in the middle of the text
  xtxt = gsub("[ \t]{2,}", " ", xtxt)
  xtxt = gsub('\\s+',' ',xtxt)
  # spaces at the beginning and end of texts
  xtxt = gsub("^\\s+|\\s+$", "", xtxt)
  # turns everything into lowercase
  if(lowercase == TRUE) xtxt = tolower(xtxt)
  return(xtxt)
}


visNGram <- function(x = NULL, ngrI = 2, ngrF = 3 ,nn = 20, 
                     show.table = TRUE, save.xlsx = TRUE, xlsx.name = NULL){
  # Function to display and save n-grams
  # x = vector of texts
  # ngrI = minimum value n-gram length (default = 2)
  # ngrF = maximum value of length n-grams (default = 3)
  # nn = number of n-grams to display for each length
  # show.table = logical, if TRUE display the results
  # save.xlsx = logical, if TRUE the results will be saved in a xlsx file
  # xlsx.name = if save.xlsx = TRUE, the name of the xlsx file in which to save the results, 
  #             if not indicated the default file is outNgram.xlsx
  require(ngram); require(dplyr); require(openxlsx)
  if(is.null(x) | class(x) != "character"){stop("invalid text vector")}
  if(ngrI < 2){stop("invalid n-gram initial value")}
  if(ngrF < ngrI){stop("invalid n-gram values: ngrF < ngrI")}
  ngr <- seq(from = ngrI, to = ngrF, by = 1)
  vnWrd <- sapply(x,FUN = wordcount)
  ngr <- rev(ngr)
  dfNGram <- data.frame(ngrams=as.character(),lung=as.numeric(),freq=as.numeric(),prop=as.numeric())
  for(i in 1:length(ngr)){
    minc <- ngr[i]-1
    ng <- ngram(x[vnWrd>minc], n=ngr[i])
    ngt <- get.phrasetable(ng)
    ngt <- ngt %>% arrange(-freq, ngrams)
    ngt$ngrams <- iconv(ngt$ngrams,from="UTF-8",to = "latin1",sub = "byte")
    ngt$ngrams <- trimws(ngt$ngrams)
    ngt$lung <- ngr[i]
    ngt <- ngt %>% arrange(-freq) %>% slice_head(n = nn)
    dfNGram <- bind_rows(dfNGram,ngt)
    dfNGram$edit <- NA
    if(show.table==TRUE){
      if(nrow(ngt)>=nn){nnr=nn} else {nnr=nrow(ngt)}
      cat("--------------------","\n")
      cat("n-gram n = ",ngr[i],"\n")
      print(ngt[1:nnr,])
    }
  }
  if(save.xlsx == TRUE){
    names(dfNGram)[2] <- "length"
    if(is.null(xlsx.name)==TRUE){xlsx.name <- "outNgram.xlsx"}
    write.xlsx(dfNGram,file = xlsx.name,rowNames = F,showNA = FALSE)
  }
}

corNGram <- function(x = NULL, ngrI = 2, ngrF = 6 ,nn = 20, dict=frmComposteOpSy, verbose = FALSE){
  # Function to modify the n-grams based of the dictionary defined by the user 
  # x = vector of texts 
  # ngrI = minimal value of the lenght of the n-grams (default = 2)
  # ngrF = maximal value of the lenght of the n-grams (default = 3)
  # nn = number of n-grams to visualize for every lenght
  # dict = dictionary compound form, by default frmComposteOpSy from OpeNER+mySyntIT
  #        must be made up in the format lemma,ngram, nparole
  #        there is also frmComposteOpSyTw from OpeNER+mySyntIT+twita
  #        both dictionaries must be loaded before using the function
  require(ngram); require(dplyr); require(stringr)
  dict <- dict
  if(is.null(x) | class(x) != "character"){
    message("vettore testi non valido")
    return()
  }
  if(exists("dict")==FALSE){
    message("dizionario forme composte non trovato")
    return()
  }
  if(ngrI < 2){
    message("valore iniziale n-grammi non valido")
    return()
  }
  if(ngrF < ngrI){
    message("valori n-grammi non validi")
    return()
  }
  ngr <- seq(from = ngrI, to = ngrF, by = 1)
  vnWrd <- numeric()
  for(i in 1:length(x)){
    vnWrd[i] <- wordcount(x[i])
  }
  ngr <- rev(ngr)
  vout <- c()
  dfmod <- data.frame(orig=character(),modifica=character(),stringsAsFactors = F)
  for(i in 1:length(ngr)){
    if(verbose==T) print(ngr[i])
    minc <- ngr[i]-1
    ng <- ngram(x[vnWrd>minc], n=ngr[i])
    ngt <- get.phrasetable(ng)
    ngt <- ngt %>% arrange(-freq, ngrams)
    ngt$ngrams <- iconv(ngt$ngrams,from="UTF-8",to = "latin1",sub = "byte")
    ngt$ngrams <- str_trim(ngt$ngrams)
    ctrl <- merge(ngt,dict,by.x = "ngrams",by.y = "ngram")
    if(nrow(ctrl)>0){
      for(kk in 1:nrow(ctrl)){
        x <- gsub(ctrl[kk,"ngrams"],ctrl[kk,"lemma"],x)
        dfmod[nrow(dfmod)+1,] <- c(ctrl[kk,"ngrams"],ctrl[kk,"lemma"])
      }
    }
  }
  if(verbose == TRUE){
    cat("--------------------","\n")
    cat("modifiche apportate","\n")
    dfmod <- dfmod %>%
      group_by(orig,modifica) %>%
      summarise(n=n())
    print(as.data.frame(dfmod))
  }
  return(x)
}

corMultWord <- function(x = NULL, xlsx.file = NULL, replace.char = "_"){
  # read the xlsx file produced by visNGram or multWordPOS functions and
  # replace multiwords defined in the field 'edit' of the xlsx file
  # if in edit "=" the spaces will be replaced with the replace.char
  # otherwise with the string reported in edit
  # x = vector of texts
  # xlsx.file = xlsx file produced by visNGram function with edit to multiwords
  # replace.char = replacement chararacter to use 
  if(is.null(x) | is.null(xlsx.file)){stop("invalid argument")}
  if(class(x) != "character" | class(xlsx.file) != "character"){stop("invalid argument")}
  require(openxlsx); require(dplyr)
  corrw <- read.xlsx(xlsx.file,sheet = 1)
  if(class(corrw) != "data.frame"){stop("invalid xlsx file")}
  corr1 <- corrw %>% arrange(-length) %>% filter(!is.na(edit)) %>% arrange(-freq)
  if(nrow(corr1)>0){
    for(i in 1:nrow(corr1)){
      if(corr1[i,"edit"]=="="){
        xcor <- gsub(" ",replace.char,corr1[i,1])
      } else {
        xcor <- corr1[i,"edit"]
      }
      x <- gsub(corr1[i,1],xcor,x)
    }
  }
  return(x)
}


corFrmComp <- function(vText=NULL, correzioni=NULL){ 
  # function for the correction of compound forms
  # Basically it effectuate the gsub of correction after visNGram
  # vText = vector of texts on which doing the correction 
  # correzioni = vector of even lenght in which there are the correction to make 
  #              forma.da.correggere.1, forma.corretta.1, forma.da.correggere.2, forma.corretta.2,...
  #              if the forma.corretta needs only the substitution of spaceswith _, then  semplicemente la sostiuzione degli spazi con _
  #              forma.corretta can be indentified with NA
  # Example: correz <- c("buona salute",NA, "carta di credito","carta_credito","partita iva", NA)
  # tw$txt <- corFrmComp(vText = tw$txt, correzioni = correz)
  if(is.null(vText) | is.null(correzioni)){
    message("mancano vettore testi o vettere forme da correggere")
    return()
  }
  if(length(correzioni) %% 2 != 0){
    message("la lunghezza del vettore correzioni non è corretta")
    return()
  }
  from <- correzioni[seq(from=1,to=length(correzioni),by = 2)]
  to <- correzioni[seq(from=2,to=length(correzioni),by = 2)]
  for(i in 1:length(from)){
    if(to[i]=="" | is.na(to[i])){
      to[i] <- gsub(" ","_",from[i])
    }
    vText <- gsub(from[i],to[i],vText)
  }
  return(vText)
}


multWordPOS <- function(x = NULL, model = "italian-isdt", n = 100, save.xlsx = TRUE, xlsx.name = NULL){
  # function to identify multiwords using pos-tagging
  # x = vector of texts
  # model = udpipe Pre-trained model 
  #         english models: english-ewt, english-gum, english-lines, english-partut,
  #         italian models: italian-isdt, italian-partut, italian-postwita, italian-twittiro, italian-vit
  # save.xlsx = logical, if TRUE the results will be saved in a xlsx file
  # xlsx.name = if save.xlsx = TRUE, the name of the xlsx file in which to save the results, 
  #             if not indicated the default file is outNgramPOS.xlsx
  require(udpipe); require(dplyr); require(openxlsx)
  if(is.null(x) | is.null(model)){stop("invalid argument")}
  if(class(x) != "character" | class(model) != "character"){stop("invalid argument")}
  out <- udpipe(x, object = model)
  AA2 <- bind_cols(tok=txt_nextgram(x=out$token, n = 2, sep = " "),
                   upo=txt_nextgram(x=out$upos, n = 2, sep = " ")) %>% 
    filter(upo %in% c("NOUN NOUN","ADJ NOUN","PROPN PROPN")) %>% 
    group_by(tok,upo) %>% summarise(nn=n(), .groups = 'drop') %>% arrange(-nn)
  AA3 <- bind_cols(tok=txt_nextgram(x=out$token, n = 3, sep = " "),
                   upo=txt_nextgram(x=out$upos, n = 3, sep = " ")) %>% 
    filter(upo %in% c("NOUN NOUN NOUN","ADJ NOUN NOUN","ADJ ADJ NOUN","PROPN PROPN PROPN",
                      "NOUN ADP NOUN","PROPN ADP PROPN")) %>% 
    group_by(tok,upo) %>% summarise(nn=n(), .groups = 'drop') %>% arrange(-nn)
  results <- bind_rows(AA2,AA3) %>% arrange(-nn) %>% slice_head(n=n) 
  colnames(results) <- c("tokens","pos","freq")
  results$edit <- NA
  if(save.xlsx==TRUE){
    if(is.null(xlsx.name)==TRUE){xlsx.name <- "outNgramPOS.xlsx"}
    write.xlsx(as.data.frame(results),file = xlsx.name,row.names = F,showNA = FALSE)
  }
  return(results)
}


lemmaUDP <- function(x = NULL, 
                     model = NULL, 
                     doc_id = NULL, 
                     stopw = tm::stopwords("italian"), 
                     userstopw=NULL,verbose=FALSE){
  # function for lemmatization with UDpipe.
  # returns a data frame in CoNLL-U format. 
  # with the addition of the STOP field identifying stopwords.
  # x = vector of texts/documents in UTF-8 format
  # model = lemmatization model
  # doc_id = document identifier
  # stopw = language stopwords list
  #         stopw = NULL to not report stopwords
  # userstopw = user-defined stopwords list.
  require(udpipe)
  if(is.null(x)){message("missing text vector");return()}
  if(is.null(model)){message("missing model");return()}
  if(class(x) != "character"){message("vector x is not character type");return()}
  if(class(model) != "udpipe_model"){message("invalid model");return()}
  if(is.null(doc_id)){doc_id <- 1:length(x)}
  if(!is.null(userstopw)){
    stopw <- c(stopw,userstopw)
  }
  xx <- udpipe_annotate(model, x = x, doc_id = doc_id,trace = verbose)
  xx <- as.data.frame(xx)
  xx$STOP <- ifelse(tolower(xx$lemma) %in% tolower(stopw) | tolower(xx$token) %in% tolower(stopw),TRUE,FALSE)
  return(xx)
}


search_reddit <- function(q = NULL, subreddit = NULL, sortby = "relevance",
                          periodo = "year", lang = "it"){
  # scarica i thread e i commenti relativi ad una query
  # q         query di ricerca
  # subreddit se non NULL, limita la ricerca al subreddit indicato
  # sortby    ordinamento dei thread ("relevance", "comments", "new", "hot", "top")
  # periodo   periodo di interesse ("hour", "day", "week", "month", "year", "all")
  # lang      lingua da selezionare "it","en", NULL seleziona tutti thread e commenti
  # deve essere stata caricata la funzione cleanText in utility.R 
  require(dplyr,quietly = T,warn.conflicts = F)
  require(RedditExtractoR,quietly = T,warn.conflicts = F)
  require(cld2,quietly = T,warn.conflicts = F)
  require(cld3,quietly = T,warn.conflicts = F)
  if(is.null(q)) stop("manca la query")
  # ------ ricerca dei thread ------
  if(!is.null(subreddit)){
    thre <- find_thread_urls(keywords = q ,
                             subreddit=subreddit,
                             sort_by=sortby,
                             period=periodo)
  } else {
    thre <- find_thread_urls(keywords = q ,
                             sort_by=sortby,
                             period=periodo)
  }
  if(class(thre) != "data.frame") stop("No thread found")
  cat(paste("trovati",nrow(thre),"thread"),"\n")
  thre <- thre %>% mutate(ID=1:nrow(thre),.before=1)
  nt = nrow(thre)
  pb <- txtProgressBar(min = 0, max = nt, style = 3)
  # ------ download dei thread e dei commenti ------
  for(i in 1:nrow(thre)){
    setTxtProgressBar(pb, i)
    tmp.cmmRDT <- tryCatch(get_thread_content(thre$url[i]),
                           error = function(e) e,
                           warning = function(w) w)
    if(!"condition" %in% class(tmp.cmmRDT)){
      # se sono presenti dei commenti
      if(!is.null(tmp.cmmRDT$comments)){
        dfcommenti <- left_join(tmp.cmmRDT$comments,
                                tmp.cmmRDT$threads %>% 
                                  arrange(timestamp) %>% 
                                  select(url,subreddit) %>% 
                                  mutate(id_thread=i),
                                by="url")
        dftmp <- bind_rows(tmp.cmmRDT$threads %>% 
                             mutate(tipo="thread",.before=1) %>% 
                             mutate(id_thread=i,.after=1),
                           dfcommenti %>% 
                             rename(text=comment) %>% 
                             mutate(tipo="comment",.before=1)) %>% 
          as_tibble() %>% 
          mutate(id=1:(nrow(tmp.cmmRDT$threads)+nrow(dfcommenti)),.before=1) %>% 
          mutate(created_at=as.POSIXct(timestamp,origin = "1970-01-01"),.after=timestamp) %>% 
          mutate(comment_id=as.character(comment_id)) %>% 
          mutate(txt=paste(title,text,collapse = " "))
      } else {
        # se non ci sono commenti
        dftmp <- tmp.cmmRDT$threads %>% mutate(tipo="thread",.before=1) %>% 
          mutate(id_thread=i,.after=1) %>% 
          as_tibble() %>% 
          arrange(id_thread,timestamp) %>% 
          mutate(id=1:(nrow(tmp.cmmRDT$threads)),.before=1) %>% 
          mutate(created_at=as.POSIXct(timestamp,origin = "1970-01-01"),.after=timestamp)
      }
      if(i==1){
        dfThCm2 <- dftmp
      } else {
        dfThCm2 <- bind_rows(dfThCm2,dftmp)
      }
    }
  }
  # ------ selezione dei testi in funzione di una lingua ------
  if(!is.null(lang)){
    # creo un campo txt con l'unione dei titoli e dei testi
    dfThCm2 <- dfThCm2 %>% mutate(txt=ifelse(is.na(title),text,
                                             ifelse(is.na(text),title,paste(title,text,sep = " "))))
    # individuo la lingua dei testi con le due librerie
    cld2_vec = cld2::detect_language(text = cleanText(dfThCm2$txt))
    cld3_vec = cld3::detect_language(text = cleanText(dfThCm2$txt))
    dfl = data.frame(id=1:length(cld2_vec),cl2=cld2_vec,cl3=cld3_vec)
    # se almeno una delle due lingue individuate è la lingua indicata con lang, 
    # allora il record viene selezionato
    idlang <- dfl %>% filter(cl2==lang | cl3==lang) %>% pull(id)
    dfThCmL2 <- dfThCm2 %>% mutate(id=1:nrow(dfThCm2),.before = 1) %>% filter(id %in% idlang)
    cat(paste("--> ",nrow(dfThCm2)-nrow(dfThCmL2)," casi in ligua diversa da: ",lang,sep=""))
  } else {
    dfThCmL2 <- dfThCm2 %>% mutate(id=1:nrow(dfThCm2),.before = 1) %>% 
      mutate(txt=ifelse(is.na(title),text,
                        ifelse(is.na(text),title,paste(title,text,sep = " "))))
  }
  return(dfThCmL2)
}

search_reddit2 <- function(q = NULL, subreddit = NULL, sortby = "relevance",
                           periodo = "year", lang = "it"){
  # scarica i thread e i commenti relativi ad una query
  # a differenza di search_reddit consente di scaricare i thread da un subreddit senza keywords
  # e il campo txt è ricodificato in UTF-8
  # q         query di ricerca (keywords)
  # subreddit se non NULL, limita la ricerca al subreddit indicato
  # sortby    ordinamento dei thread ("relevance", "comments", "new", "hot", "top")
  # periodo   periodo di interesse ("hour", "day", "week", "month", "year", "all")
  # lang      lingua da selezionare "it","en", NULL seleziona tutti thread e commenti
  # deve essere stata caricata la funzione cleanText in utility.R 
  require(dplyr,quietly = T,warn.conflicts = F)
  require(RedditExtractoR,quietly = T,warn.conflicts = F)
  require(cld2,quietly = T,warn.conflicts = F)
  require(cld3,quietly = T,warn.conflicts = F)
  if(is.null(q) & is.null(subreddit)) stop("manca query e subreddit")
  conv.Txt <- function(txt=NULL){
    txte <- c()
    for(i in 1:length(txt)){
      if(Encoding(txt[i])=="unknown"){
        txte[i] = iconv(txt[i],from="latin1",to="UTF-8")
      } else {
        txte[i] = txt[i]
      }
    }
    return(txte)
  }
  
  # ------ ricerca dei thread ------
  if(!is.null(subreddit)){
    if(is.null(q)){
      thre <- find_thread_urls(subreddit=subreddit,
                               sort_by="top",
                               period=periodo)
    } else {
      thre <- find_thread_urls(keywords = q ,
                               subreddit=subreddit,
                               sort_by=sortby,
                               period=periodo)
    }
  } else {
    thre <- find_thread_urls(keywords = q ,
                             sort_by=sortby,
                             period=periodo)
  }
  if(class(thre) != "data.frame") stop("No thread found")
  cat(paste("trovati",nrow(thre),"thread"),"\n")
  thre <- thre %>% mutate(ID=1:nrow(thre),.before=1)
  nt = nrow(thre)
  pb <- txtProgressBar(min = 0, max = nt, style = 3)
  # ------ download dei thread e dei commenti ------
  for(i in 1:nrow(thre)){
    setTxtProgressBar(pb, i)
    tmp.cmmRDT <- tryCatch(get_thread_content(thre$url[i]),
                           error = function(e) e,
                           warning = function(w) w)
    if(!"condition" %in% class(tmp.cmmRDT)){
      # se sono presenti dei commenti
      if(!is.null(tmp.cmmRDT$comments)){
        dfcommenti <- left_join(tmp.cmmRDT$comments,
                                tmp.cmmRDT$threads %>% 
                                  arrange(timestamp) %>% 
                                  select(url,subreddit) %>% 
                                  mutate(id_thread=i),
                                by="url")
        dftmp <- bind_rows(tmp.cmmRDT$threads %>% 
                             mutate(tipo="thread",.before=1) %>% 
                             mutate(id_thread=i,.after=1),
                           dfcommenti %>% 
                             rename(text=comment) %>% 
                             mutate(tipo="comment",.before=1)) %>% 
          as_tibble() %>% 
          mutate(id=1:(nrow(tmp.cmmRDT$threads)+nrow(dfcommenti)),.before=1) %>% 
          mutate(created_at=as.POSIXct(timestamp,origin = "1970-01-01"),.after=timestamp) %>% 
          mutate(comment_id=as.character(comment_id)) %>% 
          mutate(txt=paste(title,text,collapse = " "))
      } else {
        # se non ci sono commenti
        dftmp <- tmp.cmmRDT$threads %>% mutate(tipo="thread",.before=1) %>% 
          mutate(id_thread=i,.after=1) %>% 
          as_tibble() %>% 
          arrange(id_thread,timestamp) %>% 
          mutate(id=1:(nrow(tmp.cmmRDT$threads)),.before=1) %>% 
          mutate(created_at=as.POSIXct(timestamp,origin = "1970-01-01"),.after=timestamp)
      }
      if(i==1){
        dfThCm2 <- dftmp
      } else {
        dfThCm2 <- bind_rows(dfThCm2,dftmp)
      }
    }
  }
  # ------ selezione dei testi in funzione di una lingua ------
  if(!is.null(lang)){
    # creo un campo txt con l'unione dei titoli e dei testi
    dfThCm2 <- dfThCm2 %>% mutate(txt=ifelse(is.na(title),text,
                                             ifelse(is.na(text),title,paste(title,text,sep = " "))))
    # individuo la lingua dei testi con le due librerie
    dfThCm2$txt <- conv.Txt(dfThCm2$txt)
    cld2_vec = cld2::detect_language(text = cleanText(dfThCm2$txt))
    cld3_vec = cld3::detect_language(text = cleanText(dfThCm2$txt))
    dfl = data.frame(id=1:length(cld2_vec),cl2=cld2_vec,cl3=cld3_vec)
    # se almeno una delle due lingue individuate è la lingua indicata con lang, 
    # allora il record viene selezionato
    idlang <- dfl %>% filter(cl2==lang | cl3==lang) %>% pull(id)
    dfThCmL2 <- dfThCm2 %>% mutate(id=1:nrow(dfThCm2),.before = 1) %>% filter(id %in% idlang)
    cat(paste("--> ",nrow(dfThCm2)-nrow(dfThCmL2)," casi in ligua diversa da: ",lang,sep=""))
  } else {
    dfThCmL2 <- dfThCm2 %>% mutate(id=1:nrow(dfThCm2),.before = 1) %>% 
      mutate(txt=ifelse(is.na(title),text,
                        ifelse(is.na(text),title,paste(title,text,sep = " "))))
    dfThCm2$txt <- conv.Txt(dfThCm2$txt)
  }
  return(dfThCmL2)
}


reddit.users.info <- function(users=NULL){
  # funzione per scaricare info su utenti reddit
  if(is.null(users)) stop("manca users")
  flagini = 1
  nu <- 0
  user_content <- list()
  for(j in 1:length(users)){
    x <- try(get_user_content(users = users[j]))
    if(class(x)!="try-error"){
      nu <- nu+1
      for(i in 1:length(x)){
        tmp <- as.data.frame(x[[i]]$about)
        if(!is.null(x[[i]]$threads)) {n.thread=nrow(x[[i]]$threads)} else {n.thread=0}
        if(!is.null(x[[i]]$comments)) {n.comments=nrow(x[[i]]$comments)} else {n.comments=0}
        tmp <- tmp %>% mutate(n.thread=n.thread,n.comments=n.comments)
        if(flagini==1){
          dfAuths=tmp
          flagini=0
          comments <- x[[i]]$comments
          threads <- x[[i]]$threads
          user_content <- x
        } else {
          dfAuths=bind_rows(dfAuths,tmp)
          comments <- bind_rows(comments,x[[i]]$comments)
          threads <- bind_rows(threads,x[[i]]$threads)
          user_content[length(user_content)+1] <- x
        }
      }
    } else {
      print(paste("error",users[j]))
    }
  }
  dfAuths <- dfAuths %>% mutate(created_at=as.POSIXct(timestamp,origin="1970-01-01"),.after=timestamp)
  names(user_content) <- dfAuths$name
  res <- list(user_content = user_content, threads=threads, comments=comments, user_info = dfAuths)
  return(res)
}

# funzione per trasformare una data in formato testo in timestamp
data2timestamp <- function(x){
  tstmp=as.numeric(difftime(x,"1970-01-01",units = "sec",tz = "UTC"))
  return(tstmp)
}
# funzione per trasformare timestamp in una data
timestamp2data <- function(x){
  return(as.POSIXct(x,origin="1970-01-01",tz = "UTC"))
}

# funzione per determinare se la lingua di un testo è quella desiderata
det.lang.2 <- function(x = NULL, lang = "it", livello = 1){
  # x       vettore di testi
  # lang    lingua da selezionare
  # livello valore da 0 a 2, 
  #         ad esempio: 2=entrambi i criteri presentano un valore uguale a lang
  #                     1=almeno uno dei criteri ha valore uguale a lang
  # restituisce un data.frame dove il campo lang ha valori TRUE/FALSE, 
  # TRUE significa che quel testo è nella lingua selezionata 
  # con un livello pari a quello fissato
  require(cld2,quietly = T,warn.conflicts = F)
  require(cld3,quietly = T,warn.conflicts = F)
  if(is.null(x)) stop("parametri insufficienti")
  if(!livello %in% 0:2) stop("livello non valido")
  cld2_vec = cld2::detect_language(text = x)
  cld3_vec = cld3::detect_language(text = x)
  dfl=data.frame(id=1:length(cld2_vec),cl2=cld2_vec,cl3=cld3_vec)
  dfl[is.na(dfl)] <- "xx"
  dflL=dfl %>% mutate_at(2:3,~.==lang) %>% 
    mutate(value = rowSums(across(cl2:cl3)))
  dflL = dflL %>% mutate(lang = value>=livello)
  return(dflL)
}

search.Pushshift <- function(q=NULL,iniz=NULL,fine=NULL,verbose=F){
  # cerca post reddit (thread e commenti) in modo ricorsivo  su un periodo di tempo
  # q       query di ricerca
  # iniz    inizio del periodo su cui effettuare la ricerca, data in formato testo YYYY-MM-DD ("2020-01-01")
  # fine    fine del periodo su cui effettuare la ricerca, sempre in formato testo
  # verbose se TRUE visualizza i sotto-periodi su cui viene effettuata la ricerca
  require(dplyr)
  require(pushshiftR)
  if(is.null(q)) stop("missing query")
  # --------------------
  cerca.Pushshift <- function(q=NULL,iniz=NULL,fine=NULL,tipo=NULL,verbose=F){
    # funzione usata per ricercare ricursivamente submission o comment per coprire il periodo after-before
    if(!is.null(iniz)){inizz=data2timestamp(iniz)} else {inizz=NULL}
    if(!is.null(fine)){finez=data2timestamp(fine)} else {finez=NULL}
    tmp1 <- tryCatch(getPushshiftData(postType = tipo,
                                      q = q,
                                      after=inizz,
                                      before=finez,
                                      size=1000),
                     error = function(e) e,
                     warning = function(w) w)
    if("error" %in% class(tmp1)) stop("error in search")
    dfComm <- tmp1
    cat(paste("0",as.Date(timestamp2data(max(tmp1$created_utc))),as.Date(timestamp2data(min(tmp1$created_utc))),nrow(tmp1),sep = " - "),"\n")
    if(!is.null(iniz)){
      nn=0
      if(as.Date(timestamp2data(min(tmp1$created_utc)))>as.Date(iniz)){
        while(as.Date(timestamp2data(min(tmp1$created_utc)))>as.Date(iniz)){
          nn=nn+1
          if(verbose==T){
            cat(paste(nn,as.Date(timestamp2data(min(tmp1$created_utc))),sep=" - ")," - ")
          }
          tmp2 <- tryCatch(getPushshiftData(postType = tipo,
                                            q = q,
                                            after=data2timestamp(iniz)-(60*60*24),
                                            before=min(tmp1$created_utc),
                                            size=1000),
                           error = function(e) e,
                           warning = function(w) w)
          if("error" %in% class(tmp2)) break
          if(verbose==T){
            cat(paste(as.Date(timestamp2data(max(tmp2$created_utc))),stringr::str_pad(nrow(tmp2), 6, pad = " "),sep=" - "))
            cat("\n")
          }
          dfComm <- bind_rows(dfComm,tmp2)
          tmp1 <- tmp2
        } 
      }
      dfComm <- dfComm %>% filter(as.Date(timestamp2data(dfComm$created_utc))>=as.Date(iniz))
    }
    return(dfComm)
  }
  # --------------------
  tmpT <- cerca.Pushshift(q = q,iniz = iniz,fine = fine,tipo = "submission",verbose = verbose)
  cat("\n")
  tmpC <- cerca.Pushshift(q = q,iniz = iniz,fine = fine,tipo = "comment",verbose = verbose)
  # ------- aggrego i risultati delle ricerche dei thread e dei comment
  x.tot <- bind_rows(
    tmpT %>% mutate(tipo="thread",.before=1) %>% 
      select(tipo, id, created_utc, subreddit, author, title, 
             text = selftext,         # testo del thread
             num_comments, score) %>% 
      mutate(created_at = timestamp2data(created_utc),.after=created_utc),
    tmpC %>% mutate(tipo="comment",.before=1) %>% 
      select(tipo, created_utc, subreddit, author,
             text=body,               # testo del commento
             score) %>% 
      mutate(created_at = timestamp2data(created_utc),.after=created_utc)) %>% 
    arrange(created_utc) %>% 
    mutate(txt=ifelse(is.na(title),text,
                      ifelse(is.na(text),title,paste(title,text,sep = " "))))
  # -------
  return(x.tot)
}

# function to get the timeline of a vector of users
# with the complete building of the user object
# not seen in the lesson
my_get_timeline <- function(user=NULL,n=100){
  require(rtweet)
  if(is.null(user)) stop("manca user")
  if(length(user)>1){
    lst.x <- list()
    lst.x.us <- list()
    for(i in 1:length(user)){
      print(user[i])
      x <- get_timeline(user = user[i],n=n)
      x.us <- users_data(x)
      lst.x[[i]] <- x
      lst.x.us[[i]] <- x.us
    }
    for(i in 1:length(lst.x)){
      if(i==1){
        df.tw <- lst.x[[i]]
        df.us <- lst.x.us[[i]]
      } else {
        df.tw <- bind_rows(df.tw,lst.x[[i]])
        df.us <- bind_rows(df.us,lst.x.us[[i]])
      }
    }
  } else {
    df.tw <- get_timeline(user = user,n=n)
    df.us <- users_data(df.tw)
  }
  attr(df.tw,"users") <- df.us
  return(df.tw)
}


make.tweet.df <- function(x=NULL){
  # function to create integrated tweets-users dataframe, and extract hashtahs, mentions and urls
  # x = tweet object (from get_timeline or search_tweets)
  # binding tweets with users data
  require(rtweet)
  x <- bind_cols(x,
                 users_data(x) %>% 
                   select(user_id=id_str,name,screen_name,location,   # selection of fields from user object
                          user_created_at=created_at,followers_count, # can be added others fields
                          friends_count,statuses_count,verified))
  # setting up new fields
  x <- x %>% mutate(hashtags=NA,
                    user_mentions=NA,
                    urls=NA,
                    expanded_url=NA,
                    is_retweet=NA,
                    retweet_id_str=NA,
                    retweet_created_at=NA,
                    retweet_user_screen_name=NA,
                    retweet_user_id_str=NA)
  # for each row of the tweets dataframe
  for(k in 1:nrow(x)){
    if(class(x$entities[[k]])=="list"){
      x$hashtags[k] <- list(x$entities[[k]][[1]]$text)
      x$user_mentions[k] <- list(x$entities[[k]][[3]]$screen_name)
      x$urls[k] <- list(x$entities[[k]][[4]]$url)
      x$expanded_url[k] <- list(x$entities[[k]][[4]]$expanded_url)
    }
    # I verify that retweeted_status is a dataframe (if search with include_rts = FALSE it is not)
    if(class(x$retweeted_status[[k]])=="data.frame"){
      # if it is a dataframe and id_str is not NA then it is a retweet
      x$is_retweet[k] <- ifelse(is.na(x$retweeted_status[[k]]$id_str),FALSE,TRUE)
      x$retweet_id_str[k] <- x$retweeted_status[[k]]$id_str[1]
      x$retweet_created_at[k] <- x$retweeted_status[[k]]$created_at[1]
      x$retweet_user_screen_name[k] <- x$retweeted_status[[k]]$user$screen_name[1]
      x$retweet_user_id_str[k] <- x$retweeted_status[[k]]$user$id_str[1]
    } else {
      x$is_retweet[k] <- FALSE
    }
  }
  attr(x, "users") <- NULL
  return(x)
}

