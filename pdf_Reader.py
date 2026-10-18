#goal is to make make a search tool for a pdf reader.
import spacy
import keyword_spacy
from pypdf import PdfReader

nlp=spacy.load("en_core_web_md")

nlp.add_pipe("keyword_extractor", last=True, config={"top_n": 10, "min_ngram": 3, "max_ngram": 3, "strict": True})


def chunking(sentences, chunk_size):
    chunks=[]
    for i in range(0, len(sentences), chunk_size):
        chunk="".join(str(sentence) for sentence in sentences[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# Components that lemma/vector/sentence-split work doesn't need. Passing these to
# nlp.pipe(disable=...) skips them per batch instead of paying for the full pipeline.
LIGHT = ["parser", "ner", "keyword_extractor"]


def _clean(word):
    return word.lower().strip(" ").strip(".").strip(",").strip("!")


def first_page_text(pdf_source):
    return PdfReader(pdf_source).pages[0].extract_text() or ""


def extract_chunks(pdf_source, chunk_size=2):
    """Extract keyworded chunks from a PDF. pdf_source: path or file-like object."""
    pdf=PdfReader(pdf_source)
    pages=[page.extract_text() for page in pdf.pages]
    # sentence splitting needs the parser but not keyword extraction
    tokens=[sent for doc in nlp.pipe(pages, disable=["keyword_extractor"]) for sent in doc.sents]

    chunks=chunking(tokens, chunk_size)

    chunk_data=[]
    global_keywords={}

    # one batched pass for keywords, one for lemmas (each unique keyword lemmatized once)
    chunk_docs=list(nlp.pipe(chunks))
    kept=[[(_clean(w), f, float(s)) for w,f,s in doc._.keywords if float(s)>0.5] for doc in chunk_docs]
    unique_words=sorted({w for k in kept for w,_,_ in k})
    lemmas={w:" ".join(t.lemma_ for t in d) for w,d in zip(unique_words, nlp.pipe(unique_words, disable=LIGHT))}

    #getting keywords from chunks, lemmatizing them, if lemmas match: dedup
    for chunk, chunk_keywords in zip(chunks, kept):
        data={}
        data["context"]=chunk
        data["meta"]=[]
        for word,freq,score in chunk_keywords:
            final_result=lemmas[word]
            if final_result in global_keywords:
                global_keywords[final_result]["freq"]+=freq
                if global_keywords[final_result]["score"]<score:
                    global_keywords[final_result]["score"]=score
            else:
                global_keywords[final_result]={"score":score, "freq":freq}
            data["meta"].append({
                "keyword":final_result,
                "score":score,
                "freq":freq
            })
        chunk_data.append(data)

    return chunk_data, global_keywords

