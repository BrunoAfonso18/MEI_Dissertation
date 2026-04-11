from pyabsa import AspectTermExtraction as ATEPC

# Load a pre-trained multilingual checkpoint
extractor = ATEPC.AspectExtractor(
    checkpoint="english",  # or "english" for English-only
    auto_device=True,           # uses GPU if available
)

reviews = [
    "The battery life is amazing but the screen is disappointing.",
    "Great food, but the service was terribly slow.",
    "The camera quality is outstanding and the price is very reasonable.",
    # "A bateria é razoável mas o ecrã é demasiado pequeno!",
    # "A performance é muito rápida e o arranque é quase instantâneo.",
    # "O computador não é muito silencioso.",
]

results = extractor.predict(
    reviews,
    save_result=False,
    print_result=True,
    ignore_error=True,
)

# Parse results
for result in results:
    print(f"\nReview: {result['sentence']}")
    for aspect, opinion, sentiment in zip(
        result["aspect"],
        result["opinion"],
        result["sentiment"],
    ):
        print(f"  Aspect Term  : {aspect}")
        print(f"  Opinion Term : {opinion}")
        print(f"  Polarity     : {sentiment}")