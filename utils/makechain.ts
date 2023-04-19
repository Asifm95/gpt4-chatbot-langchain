import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
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
      .map((result) => result[0]);
  };
  return retriever;
};

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If the follow up question is not related to the conversation, dont change it.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;

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

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    createRetriever(vectorstore),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
