import torch
from med1_llm import BigramLanguageModel

print(1)
# Load the saved model
model_path = "model_state_dict.pth"  # Path to the saved model
loaded_model = torch.load(model_path)
print(2)

# Set the model to evaluation mode
loaded_model.eval()

print(3)

# Define a function to generate responses
def generate_response(context, max_tokens):
    with torch.no_grad():
        context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        response = loaded_model.generate(context, max_tokens)
    return response.tolist()[0]

t=1
while t:

    x = input("enter prompt")

    # Example usage
    context = encode(x) # Start with an empty context
    max_tokens = 1000  # Maximum number of tokens to generate
    response = generate_response(context, max_tokens)

    # Decode the generated response
    decoded_response = decode(response)
    print("Generated Response: ", decoded_response)

    t = int(input("again 1, end 0"))
