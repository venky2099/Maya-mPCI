import sys, torch
sys.path.insert(0, '.')

def main():
    from maya_cl.utils.config import T_STEPS, A_MANAS
    from maya_cl.network.backbone import MayaPranaNet
    from maya_cl.encoding.poisson import PoissonEncoder
    from maya_cl.benchmark.split_cifar100 import get_task_loaders
    from maya_cl.plasticity.mpci import extract_spike_matrix, normalised_lzc, compute_mpci

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MayaPranaNet(use_orthogonal_head=False, a_manas=A_MANAS).to(device)
    model.load_state_dict(torch.load("results/phase1_reactive.pt", map_location=device, weights_only=False))
    model.eval()

    encoder = PoissonEncoder(T_STEPS)
    _, test_loader = get_task_loaders(0)
    for images, _ in test_loader:
        images = images[:32].to(device)
        spike_seq = encoder(images)
        break

    binary = extract_spike_matrix(model, spike_seq, device)
    print("Spike train length:", len(binary))
    print("Nonzero spikes:", binary.sum(), "/", len(binary))
    print("Spike density:", round(float(binary.mean()), 4))
    print("Baseline LZC:", round(normalised_lzc(binary), 6))

    result = compute_mpci(model, spike_seq, device, n_perturbations=3, seed_base=42)
    print("mPCI mean:", round(result["mpci_mean"], 6))
    print("mPCI std: ", round(result["mpci_std"], 6))
    print("Spike density:", round(result["spike_density"], 4))

if __name__ == "__main__":
    main()
