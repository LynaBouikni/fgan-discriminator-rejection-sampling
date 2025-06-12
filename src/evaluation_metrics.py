real_dataloader = DataLoader(real_dataset, batch_size=64, shuffle=True)
gen_dataloader = DataLoader(gen_dataset, batch_size=64, shuffle=True)

if epoch % evaluation_interval == 0:
    precision, recall = compute_precision_recall(G, real_dataloader, gen_dataloader, device)
    print(f'Epoch {epoch} - Precision: {precision}, Recall: {recall}')
