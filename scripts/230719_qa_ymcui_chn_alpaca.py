from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


import argparse

def arg():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    parser.add_argument('--model_path', type=str, help="Path of language models", default="/home/ubuntu/models")
    


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path',
    #                     type=str,
    #                     help='Path to the model')
    # args = parser.parse_args()
    

    # Parse the arguments
    args = parser.parse_args()

    # Do something with the arguments
    model_path = args.model_path
    print(model_path)
    # print(args.accumulate(args.integers))
    
    return args


def main():
    arguments = arg()
    print(arguments)
    model_path = arguments.model_path
    template = """Question: {question}

    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager
    
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path = model_path,
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
        callback_manager=callback_manager,
        verbose=True,
        )
    
    prompt = """
    Question: A rap battle between Stephen Colbert and John Oliver
    """
    llm(prompt)
    
    
if __name__ == "__main__":
    main()
    
    
