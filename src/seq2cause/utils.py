import torch


def next_token_collate(batch, device: str = "cuda"):
    """
    Standard next token collate function to create
    batches of input_ids and attention_mask tensors for the model.

    Args:
        batch: A list of dictionaries, each containing 'input_ids' and 'attention_mask'.
        device: The target device to move the tensors to (default is 'cuda').
    """

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_ids_batch = torch.tensor(input_ids).long().to(device)
    attention_mask_batch = torch.tensor(attention_mask).long().to(device)
    # Create the new batch dictionary
    new_batch = {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
    }
    return new_batch
