import sentencepiece as spm
import plac


def main(inf, vocab_size, outf, input_sentence_size=0):
    if input_sentence_size:
        spm.SentencePieceTrainer.train(input=inf, model_prefix=outf, vocab_size=vocab_size, model_type='bpe', input_sentence_size=input_sentence_size, shuffle_input_sentence=True)
    else:
        spm.SentencePieceTrainer.train(input=inf, model_prefix=outf, vocab_size=vocab_size, model_type='bpe')


plac.call(main)
