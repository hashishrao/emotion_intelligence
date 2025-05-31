def get_user_input(prompt):
    while True:
        user_input = input(prompt + " (yes/no): ").lower()
        if user_input == "yes" or user_input == "no":
            return user_input
        else:
            print("Please enter 'yes' or 'no'.")

def calculate_dyslexia_likelihood():
    questions = [
        "Do you have difficulty recognizing familiar words quickly and accurately?",
        "Do you struggle with decoding unfamiliar words?",
        "Do you find it challenging to segment words into individual sounds?",
        "Do you have trouble with phonological awareness tasks, such as rhyming or identifying syllables?",
        "Do you frequently omit, substitute, or reverse letters or sounds when reading or writing?",
        "Do you experience difficulty with rapid naming tasks, such as naming letters or numbers quickly?",
        "Do you have trouble with reading fluency, such as reading at a slow pace with errors?",
        "Do you find it challenging to comprehend written passages or texts?",
        "Do you struggle with spelling, particularly with phonetically irregular words?",
        "Do you have a family history of reading difficulties or dyslexia?"
    ]

    yes_count = 0

    for question in questions:
        answer = get_user_input(question)
        if answer == "yes":
            yes_count += 1

    percentage = (yes_count / len(questions)) * 100

    if percentage >= 50:
        print("Based on your responses, there is a likelihood of dyslexia.")
    else:
        print("Based on your responses, dyslexia is less likely.")

calculate_dyslexia_likelihood()