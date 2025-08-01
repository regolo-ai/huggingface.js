/* Regolo.ai provider helper for Hugging Face JS client
 * Covers: chat-completion, text-generation (LLM only)
 * ©2025
 */

import { BaseConversationalTask, BaseTextGenerationTask } from "./providerHelper.js";
import type {
	ChatCompletionOutput,
	TextGenerationInput,
	TextGenerationOutput,
	TextGenerationOutputFinishReason,
} from "@huggingface/tasks";
import type { BodyParams } from "../types.js";
import { InferenceClientProviderOutputError } from "@huggingface/inference";
import { omit } from "../utils/omit.js";

interface RegoloTextCompletionOutput extends Omit<ChatCompletionOutput, "choices"> {
	id: string;
	object: string;
	created: number;
	model: string;
	choices: Array<{
		text: string;
		finish_reason: TextGenerationOutputFinishReason;
		logprobs: unknown;
		index: number;
	}>;
	usage: Array<{
		completion_tokens: number;
		prompt_tokens: number;
		total_tokens: number;
		prompt_tokens_details: unknown;
		completion_tokens_details: unknown;
	}>;
	system_fingerprint: unknown;
	kv_transfer_params: unknown;
}

const REGOLO_BASE_URL = "https://api.regolo.ai";

export class RegoloConversationalTask extends BaseConversationalTask {
	constructor() {
		super("regoloai", REGOLO_BASE_URL);
	}
}

export class RegoloTextGenerationTask extends BaseTextGenerationTask {
	constructor() {
		super("regoloai", REGOLO_BASE_URL);
	}

	private validateResponse(response: RegoloTextCompletionOutput) {
		if (
			!(
				typeof response === "object" &&
				"choices" in response &&
				Array.isArray(response?.choices) &&
				typeof response?.model === "string"
			)
		)
			throw new InferenceClientProviderOutputError("Received malformed response from Regolo AI text generation API");
	}

	override preparePayload(params: BodyParams<TextGenerationInput>): Record<string, unknown> {
		const { model, args } = params;
		const { inputs, stream, parameters } = args;

		const p =
			parameters && "max_new_tokens" in parameters
				? {
						max_tokens: parameters.max_new_tokens,
						...omit(parameters, "max_new_tokens"),
				  }
				: parameters ?? {};

		return {
			model,
			prompt: inputs,
			...(stream ? { stream: true } : {}),
			...p,
		};
	}

	override async getResponse(response: RegoloTextCompletionOutput): Promise<TextGenerationOutput> {
		console.debug(response);
		this.validateResponse(response);
		const completion = response.choices[0];
		return {
			generated_text: completion.text,
		};
	}
}
