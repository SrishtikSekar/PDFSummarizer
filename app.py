import streamlit as st
import hashlib
import tempfile
import os
from gtts import gTTS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from pymongo import MongoClient


# Connecting to MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "pdf_notes_db"
COLLECTION_NAME = "notes"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# Voice generation section
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_name = tmp_file.name + ".mp3"
        tts.save(tmp_file_name)
    return tmp_file_name


# Initialize components
st.title("📄 PDF Summarizer, Note-Taking, & Question Answering Tool")
st.sidebar.header("🔧 Configurations")
page_selection = st.sidebar.radio("Choose Analysis Mode", ["Single Page Summary", "Page Range Summary", "Overall Summary", "Question Answering"])

api_key = "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"
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
embeddings = HuggingFaceEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)


# MongoDB Functions
def save_notes_to_db(pdf_id, file_name, section_key, summary, notes):
    existing_note = collection.find_one({"pdf_id": pdf_id, "section_key": section_key})
    if existing_note:
        collection.update_one({"_id": existing_note["_id"]}, {"$set": {"summary": summary, "notes": notes}})
    else:
        collection.insert_one({"pdf_id": pdf_id, "file_name": file_name, "section_key": section_key, "summary": summary, "notes": notes})


def load_notes_from_db(pdf_id, section_key):
    result = collection.find_one({"pdf_id": pdf_id, "section_key": section_key})
    return result["notes"] if result else ""


def delete_note_from_db(pdf_id, section_key):
    collection.delete_one({"pdf_id": pdf_id, "section_key": section_key})


def load_all_pdf_entries():
    entries = collection.aggregate([{"$group": {"_id": "$pdf_id", "file_name": {"$first": "$file_name"}}}])
    return [(entry["_id"], entry["file_name"]) for entry in entries]


def load_notes_by_pdf_id(pdf_id):
    entries = collection.find({"pdf_id": pdf_id})
    return [(entry["section_key"], entry["summary"], entry["notes"]) for entry in entries]


# Generate unique PDF ID
def generate_pdf_id(file):
    file_hash = hashlib.md5(file.getvalue()).hexdigest()
    return file_hash


# Display summary and take notes
def display_summary_and_take_notes(pdf_id, section_key, summary):
    st.subheader("🔍 Summary")
    st.write(summary)

    # Display the audio
    audio_file = text_to_speech(summary)
    st.audio(audio_file, format="audio/mp3")

    # Load any saved notes for this section
    notes = load_notes_from_db(pdf_id, section_key)
    
    # Display the note-taking section
    st.subheader("📝 Take Notes")
    note_input = st.text_area("Write your notes here:", value=notes or "", key=f"note_input_area_{section_key}")
    
    # Save notes button
    if st.button("Save Note", key=f"save_{section_key}"):
        save_notes_to_db(pdf_id, file_name, section_key, summary, note_input)
        st.success("Note saved to database!")


# File upload and processing
pdf_file = st.file_uploader("📁 Upload a PDF file", type="pdf")
if pdf_file is not None:
    pdf_id = generate_pdf_id(pdf_file)
    file_name = pdf_file.name

    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            pdf_path = tmp_file.name
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
    
    if pages:
        st.success("PDF Loaded Successfully!")
    else:
        st.error("Failed to load PDF pages. Please try another file.")

    if pages:
        if page_selection == "Single Page Summary":
            st.sidebar.subheader("Single Page Settings")
            page_number = st.sidebar.number_input("Enter page number", min_value=1, max_value=len(pages), value=1, step=1)
            view = pages[page_number - 1]
            texts = text_splitter.split_text(view.page_content)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="stuff")
            
            with st.spinner("Generating Summary..."):
                summary = chain.run(docs)
            section_key = f"Page-{page_number}"
            display_summary_and_take_notes(pdf_id, section_key, summary)

        elif page_selection == "Page Range Summary":
            st.sidebar.subheader("Page Range Settings")
            start_page = st.sidebar.number_input("Start page", min_value=1, max_value=len(pages), value=1, step=1)
            end_page = st.sidebar.number_input("End page", min_value=start_page, max_value=len(pages), value=start_page, step=1)
            
            texts = []
            for page_number in range(start_page, end_page + 1):
                view = pages[page_number - 1]
                page_texts = text_splitter.split_text(view.page_content)
                texts.extend(page_texts)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="stuff")
            
            with st.spinner("Generating Summary..."):
                summary = chain.run(docs)
            section_key = f"Pages-{start_page}-to-{end_page}"
            display_summary_and_take_notes(pdf_id, section_key, summary)

        elif page_selection == "Overall Summary":
            combined_content = ''.join([p.page_content for p in pages])
            texts = text_splitter.split_text(combined_content)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="stuff")
            
            with st.spinner("Generating Overall Summary..."):
                summary = chain.run(docs)
            section_key = "Overall-Summary"
            display_summary_and_take_notes(pdf_id, section_key, summary)
        
        elif page_selection == "Question Answering":
            st.sidebar.subheader("Question Answering")
            page_number = st.sidebar.number_input("Select Page for QA", min_value=1, max_value=len(pages), value=1, step=1)
            view = pages[page_number - 1]
            texts = text_splitter.split_text(view.page_content)
            docs = [Document(page_content=t) for t in texts]
            
            # Input field for user question
            user_question = st.text_input("Enter your question:")
            if user_question:
                qa_chain = load_qa_chain(llm, chain_type="map_reduce")
                with st.spinner("Fetching Answer..."):
                    answer = qa_chain.run(question=user_question, input_documents=docs)
                st.subheader("Answer:")
                st.write(answer)
else:
    st.warning("Please upload a PDF file to get started.")

# Displaying saved notes
st.sidebar.subheader("📜 View Saved Notes")
pdf_entries = load_all_pdf_entries()
selected_pdf = st.sidebar.selectbox("Select a PDF to view notes", [None] + [f"{e[1]}" for e in pdf_entries])

if selected_pdf:
    selected_pdf_id = [e[0] for e in pdf_entries if e[1] == selected_pdf][0]
    notes_sections = load_notes_by_pdf_id(selected_pdf_id)
    
    st.header(f"📝 Notes for {selected_pdf}")
    for section_key, summary, notes in notes_sections:
        st.subheader(f"Section: {section_key}")
        st.write("Summary:")
        st.write(summary)
        st.write("Notes:")
        st.write(notes if notes else "No notes taken yet.")
        
        # Delete button for a note
        if st.button("Delete Note", key=f"delete_{section_key}"):
            delete_note_from_db(selected_pdf_id, section_key)
            st.success(f"Note for section '{section_key}' deleted.")
