import ollama

# directly use the MedGemma model for chest X-ray analysis
response = ollama.chat(model='hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M', messages=[
  {
    'role': 'user',
    'content': 'Analyze this chest X-ray. Focus on lung parenchymal abnormalities, specifically checking for infiltrates, nodules,or cavitary lesions. Provide a differential diagnosis.\n\n![chest—Xray](https://upload.wikimedia.org/wikipedia/commons/4/4f/Tuberculosis-x-ray.jpg)',
  },
])
print(response['message']['content'])
