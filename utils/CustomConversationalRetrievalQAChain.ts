import { PromptTemplate } from 'langchain';
import {
  BaseChain,
  ChainInputs,
  ConversationalRetrievalQAChain,
  LLMChain,
  loadQAStuffChain,
} from 'langchain/chains';
import { ConversationalRetrievalQAChainInput } from 'langchain/dist/chains/conversational_retrieval_chain';
import { BaseLLM } from 'langchain/dist/llms/base';
import { BaseRetriever, ChainValues } from 'langchain/dist/schema';

const question_generator_template = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const qa_template = `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:`;

const empty_context_response_generator_template = `You are a helpful AI assistant. No context is available. If the below question is a greeting, respond back politely. If the question is unrelated to Collect chat, politely respond that you can only answer questions that are related to Collect chat. Dont asnwer questions that asks "What is x?" if x is not related to Collect chat. Never break character.

Question: {question}
Helpful Answer:`;

export interface CustomConversationalRetrievalQAChainInput
  extends Omit<ChainInputs, 'memory'> {
  retriever: BaseRetriever;
  combineDocumentsChain: BaseChain;
  questionGeneratorChain: LLMChain;
  emptyContextResponseGeneratorChain?: LLMChain;
  returnSourceDocuments?: boolean;
  inputKey?: string;
}

export class CustomConversationalRetrievalQAChain
  extends ConversationalRetrievalQAChain
  implements CustomConversationalRetrievalQAChainInput
{
  outputKey = 'text';
  emptyContextResponseGeneratorChain?: LLMChain;

  constructor(fields: CustomConversationalRetrievalQAChainInput) {
    super(fields);
    if (fields.questionGeneratorChain) {
      this.emptyContextResponseGeneratorChain =
        fields.emptyContextResponseGeneratorChain;
    }
  }

  async _call(values: ChainValues): Promise<ChainValues> {
    if (!(this.inputKey in values)) {
      throw new Error(`Question key ${this.inputKey} not found.`);
    }
    if (!(this.chatHistoryKey in values)) {
      throw new Error(`chat history key ${this.inputKey} not found.`);
    }
    const question: string = values[this.inputKey];
    const chatHistory: string = values[this.chatHistoryKey];
    let newQuestion = question;
    if (chatHistory.length > 0) {
      const result = await this.questionGeneratorChain.call({
        question,
        chat_history: chatHistory,
      });
      const keys = Object.keys(result);
      if (keys.length === 1) {
        newQuestion = result[keys[0]];
      } else {
        throw new Error(
          'Return from llm chain has multiple values, only single values supported.',
        );
      }
    }
    const docs = await this.retriever.getRelevantDocuments(newQuestion);
    if (docs.length === 0 && this.emptyContextResponseGeneratorChain) {
      const result = await this.emptyContextResponseGeneratorChain.call({
        question,
        chat_history: chatHistory,
      });
      const keys = Object.keys(result);
      if (keys.length === 1) {
        return {
          [this.outputKey]: result[keys[0]],
        };
      } else {
        throw new Error(
          'Return from llm chain has multiple values, only single values supported.',
        );
      }
    }
    const inputs = {
      question: newQuestion,
      input_documents: docs,
      chat_history: chatHistory,
    };
    const result = await this.combineDocumentsChain.call(inputs);
    if (this.returnSourceDocuments) {
      return {
        ...result,
        sourceDocuments: docs,
      };
    }
    return result;
  }
  static fromLLM(
    llm: BaseLLM,
    retriever: BaseRetriever,
    options: {
      inputKey?: string;
      outputKey?: string;
      returnSourceDocuments?: boolean;
      questionGeneratorTemplate?: string;
      emptyContextResponseGeneratorTemplate?: string;
      qaTemplate?: string;
    } = {},
  ): CustomConversationalRetrievalQAChain {
    const {
      questionGeneratorTemplate,
      qaTemplate,
      emptyContextResponseGeneratorTemplate,
      ...rest
    } = options;
    const question_generator_prompt = PromptTemplate.fromTemplate(
      questionGeneratorTemplate || question_generator_template,
    );
    const qa_prompt = PromptTemplate.fromTemplate(qaTemplate || qa_template);
    const qaChain = loadQAStuffChain(llm, { prompt: qa_prompt });
    const questionGeneratorChain = new LLMChain({
      prompt: question_generator_prompt,
      llm,
    });

    let emptyContextResponseGeneratorChain: LLMChain | undefined;
    if (emptyContextResponseGeneratorTemplate) {
      const empty_context_response_generator_prompt =
        PromptTemplate.fromTemplate(
          emptyContextResponseGeneratorTemplate ||
            empty_context_response_generator_template,
        );
      emptyContextResponseGeneratorChain = new LLMChain({
        prompt: empty_context_response_generator_prompt,
        llm,
      });
    }

    const instance = new this({
      retriever,
      combineDocumentsChain: qaChain,
      questionGeneratorChain,
      emptyContextResponseGeneratorChain,
      ...rest,
    });
    return instance;
  }
}
