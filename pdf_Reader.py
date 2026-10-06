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


def extract_chunks(pdf_source, chunk_size=2):
    """Extract keyworded chunks from a PDF. pdf_source: path or file-like object."""
    pdf=PdfReader(pdf_source)
    tokens=[]
    for page in pdf.pages:
        content=nlp(page.extract_text())
        tokens.extend(list(content.sents))

    chunks=chunking(tokens, chunk_size)

    chunk_data=[]
    global_keywords={}

    #getting keywords from chunks, lemmatizing them, if lemmas match: dedup
    for chunk in chunks:
        data={}
        content=nlp(chunk)
        keywords=content._.keywords
        data["context"]=chunk
        data["meta"]=[]
        for word,freq,score in keywords:
            score=float(score)
            if(score>0.5):
                result=word.lower().strip(" ").strip(".").strip(",").strip("!")
                doc_result=nlp(result)
                final_result=" ".join(token.lemma_ for token in doc_result)
                if final_result in global_keywords:
                    global_keywords[final_result]["freq"]+=freq;
                    if global_keywords[final_result]["score"]<score:
                        global_keywords[final_result]["score"]=score;
                else:
                    global_keywords[final_result]={"score":float(score), "freq":freq}
                data["meta"].append({
                    "keyword":final_result,
                    "score":score,
                    "freq":freq
                })
        chunk_data.append(data)

    return chunk_data, global_keywords

