import gradio as gr
from transformers import pipeline
from gec.correcter import Correcter

classifier = pipeline(model="BonySmoke/gec_uk_seq2tag",
                      aggregation_strategy="simple")

examples = [
    [0.2, 0.60, 3, "на  жаль можливо навіть він нам не допоможе."],
    [0.2, 0.60, 1, "Старі часи минули, Боб; я б хотів аби вони не закінчились, щоб ми могли ще раз повечеряти тут."],
    [0.0, 0.60, 1, "ми працювали над цим проектом надзвичайно довго."]
]


def ner(prediction_confidence, prediction_delete_confidence, stages, text):
    correcter = Correcter(text,
                          min_score=prediction_confidence,
                          min_delete_score=prediction_delete_confidence,
                          classifier=classifier)
    correction = correcter.correct(stages=int(stages))
    # let's only show the first prediction as an example
    # but the text will be final
    prediction = correction["stages"][0]["prediction"]

    return {"text": text, "entities": prediction}, correction["final"]


demo = gr.Interface(
    ner,
    inputs=[
        gr.Slider(0, 1, value=0.20, step=0.1, label="Prediction Confidence"),
        gr.Slider(0, 1, value=0.60, step=0.1,
                  label="Prediction Confidence for deletes"),
        gr.Number(value=3, label="The number of correction iterations"),
        gr.Textbox(placeholder="Введіть ваше речення...")
    ],
    outputs=["highlight", "text"],
    examples=examples
)

demo.launch()
