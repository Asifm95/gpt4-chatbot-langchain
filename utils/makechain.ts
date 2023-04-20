import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { CustomConversationalRetrievalQAChain } from './CustomConversationalRetrievalQAChain';
import { CallbackManager } from 'langchain/callbacks';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { BaseRetriever } from 'langchain/dist/schema';

export const createRetriever = (
  vectorStore: PineconeStore,
  filter?: any,
): BaseRetriever => {
  const embeddings = new OpenAIEmbeddings();
  const retriever = vectorStore.asRetriever();
  retriever.getRelevantDocuments = async (query: any) => {
    const results = await vectorStore.similaritySearchVectorWithScore(
      await embeddings.embedQuery(query),
      3,
    );
    return results
      .filter(([_, score]) => score >= 0.78)
      .map(([result, score]) => ({
        ...result,
        metadata: { ...result.metadata, score },
      }));
  };
  return retriever;
};

const CONDENSE_PROMPT = `Given the following conversation and a follow up input, rephrase the follow up question to be a standalone question. If the follow up input is not related to the conversation or not a question, just return the follow up input as is.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you can only answer questions that are related to Collect chat.

{context}

If the answer is not included, follow the rules below:
1. If the question is a greeting, respond back politely.
3. If the qestion is unrelated to Collect chat, politely respond that you can only answer questions that are related to Collect chat.

Make sure the answer is in markdown format. Add line breaks when needed. Use bullet points if needed. Use bold, italics, and links if needed. Hyperlink URLs if possible. If the answer is a code snippet, use the code block markdown.

Question: """
{question}
"""

Helpful answer in markdown:`;

const EMPTY_CONTEXT_RESPONSE_PROMPT = `You are a helpful AI assistant. No context is available. If the below question is a greeting, respond back politely. If the question is unrelated to Collect chat, politely respond that you can only answer questions that are related to Collect chat. Dont asnwer questions that asks "What is x?" or "Who is x?" if x is not related to Collect chat. Never break character.

Question: {question}
Helpful Answer:`;

// const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

// Context: {context}

// If you don't know the answer or if the context is empty, just say "I'm not tuned to answer that. How else can I be of assistance?". DO NOT try to make up an answer.
// If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. If the question is a greeting, say "Hello, how can I help you?". Never break character.

// Question: {question}
// Helpful answer in markdown:`;

// const QA_PROMPT = `I want you to act as a document that I am having a conversation with. Your name is "AI Assistant". You will provide me with answers from the given info below.

// {context}

// If the answer is not included, say exactly "Hmm, I am not sure." and stop after that. Refuse to answer any question not about Collect chat. Never break character.

// Question: {question}
// Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    streaming: true,
    callbackManager: CallbackManager.fromHandlers({
      async handleLLMStart(llm, prompts, verbose) {
        console.log('Starting LLM');
        console.log('Prompt:', prompts);
        console.log('-------------------------------------------');
      },
    }),
  });

  const chain = CustomConversationalRetrievalQAChain.fromLLM(
    model,
    createRetriever(vectorstore),
    // vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      emptyContextResponseGeneratorTemplate: EMPTY_CONTEXT_RESPONSE_PROMPT,
      returnSourceDocuments: true,
    },
  );
  return chain;
};
