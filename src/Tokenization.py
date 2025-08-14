import torch

# def tokenizer(text, encode = True, mask = None, max_seq_length=32):
#     if encode:
#         out = chr(2) + text + chr(3)
        
#         if len(out) > max_seq_length:
#             out = out[:max_seq_length]
#         # ------------------------------------------------------------------------------ |
#         # Tokenizer function: Prepares text for model input when `encode=True`.          |
#         # ------------------------------------------------------------------------------ |
#         # 1. `chr(2)` → Start of Text (SOT) token.                                       |
#         #    `chr(3)` → End of Text (EOT) token.                                         |
#         #    These are added around the input text to clearly mark boundaries,           |
#         #    making it easier for the model to know where the sequence begins and ends.  |
#         #                                                                                |
#         # 2. Example:                                                                    |
#         #       text = "Hi"                                                              |
#         #       out  = chr(2) + "Hi" + chr(3)                                            |
#         #       → "[SOT]Hi[EOT]"                                                         |
#         #                                                                                |
#         # 3. Length control:                                                             |
#         #    - `max_seq_length` defines the maximum sequence length the model can take.  |
#         #    - If the combined string (SOT + text + EOT) is longer than `max_seq_length`,|
#         #      it is truncated to fit within this limit.                                 |
#         #                                                                                |
#         # Why this matters:                                                              |
#         # - Prevents sequence overflow that could break the model input size.            |
#         # - Maintains consistent input shape for batching.                               |
#         # ------------------------------------------------------------------------------ |
        
#         out = out + "".join([
#         chr(0) for _ in range(max_seq_length-len(out))   
#         ])
#         # ------------------------------------------------------------------------------ |
#         # Pads the string `out` with null characters (chr(0)) until it reaches           |
#         # `max_seq_length`. This ensures all sequences have the same length,             |
#         # which is required for efficient batching in the model.                         |
#         # ------------------------------------------------------------------------------ |
        
#         out = torch.IntTensor(
#             list(out.encode('utf-8'))
#         )
#         # ------------------------------------------------------------------------------ |
#         # Convert the input string `out` to a UTF-8 encoded bytes sequence.              |
#         # Then convert that bytes sequence into a list of integer byte values.           |
#         # Finally, convert this list into a PyTorch IntTensor for model input.           |
#         #                                                                                |
#         # Explanation:                                                                   |
#         # - `out.encode("utf-8")` converts the string to bytes using UTF-8 encoding,     |
#         #   representing characters as byte values.                                      |
#         # - `list(...)` transforms the bytes into a list of integer byte codes.          |
#         # - `torch.IntTensor(...)` creates a tensor of integers from the list,           |
#         #   which can be used as input to the Transformer or other models.               |
#         # ------------------------------------------------------------------------------ |


#         mask = torch.ones(len(out.nonzero()))
#         # ------------------------------------------------------------------------------ |
#         # Creates a mask tensor of ones for the input sequence.                          |
#         #                                                                                |
#         # Explanation:                                                                   |
#         # - `out.nonzero()` returns indices of non-zero elements in `out`.               |
#         # - `len(out.nonzero())` gives the count of those non-zero elements.             |
#         # - `torch.ones(...)` creates a tensor filled with ones of that length.          |
#         #                                                                                |
#         # This mask is typically used in attention to indicate valid token positions.    |
        
#         if len(mask) < max_seq_length:
#             mask = torch.cat(
#                 (mask, torch.zeros(max_seq_length - len(mask)))
#                 ).type(torch.IntTensor)
#         else:
#             mask = mask.type(torch.IntTensor)
#         # ------------------------------------------------------------------------------ |
#         # Pad the `mask` tensor with zeros if its length is less than `max_seq_length`.  |
#         # This ensures the mask matches the required sequence length for batching.       |
#         #                                                                                |
#         # Explanation:                                                                   |
#         # - If `mask` is shorter than `max_seq_length`, zeros are concatenated to it,    |
#         #   representing padded positions.                                               |
#         # - If `mask` is already long enough, it is simply converted to an IntTensor.    |
#         # - Padding positions (zeros) tell the model which tokens to ignore during       |
#         #   attention calculations.                                                      |
#         # - The mask is cast to `torch.IntTensor` to be compatible with model inputs.    |
#         # ------------------------------------------------------------------------------ |

#     else:
#         out = [chr(c) for c in text[1:len(mask.nonzero())-1]]
#         out = "".join(out)
#         mask = None
#         # ------------------------------------------------------------------------------ |
#         # Decode the sequence of token integers back into a string.                      |
#         #                                                                                |
#         # Explanation:                                                                   |
#         # - Converts each integer token in `text` (excluding start and end tokens)       |
#         #   back to its corresponding character using `chr()`.                           |
#         # - Uses slicing `text[1:len(mask.nonzero()) - 1]` to ignore special tokens.     |
#         # - Joins the list of characters into a complete decoded string.                 |
#         # - Resets `mask` to `None` as it is no longer needed after decoding.            |
#         # ------------------------------------------------------------------------------ |
        
#     return out, mask
        


def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        # Adding SOT and EOT tokens
        out = chr(2) + text + chr(3)
        
        # Truncate if length exceeds max_seq_length
        if len(out) > max_seq_length:
            out = out[:max_seq_length]
        
        # Add padding if needed
        out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
        
        # Encode the text
        out = torch.IntTensor(list(out.encode("utf-8")))
        
        # Create the mask
        mask = torch.ones(len(out.nonzero()))
        
        # Pad the mask to max_seq_length
        if len(mask) < max_seq_length:
            mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(torch.IntTensor)
        else:
            mask = mask.type(torch.IntTensor)
    else:
        # Decode the text
        out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
        out = "".join(out)
        mask = None

    return out, mask
