"""
Script pour convertir les anciens checkpoints vers la nouvelle architecture DuelingDQN.
L'ancienne architecture avait fc1 et fc2, la nouvelle a fc_val, fc_adv, val, adv.
"""
import torch
import argparse
import os


def convert_old_to_dueling(old_state_dict, n_actions=4):
    """
    Convertit un ancien checkpoint (fc1, fc2) vers Dueling DQN (fc_val, fc_adv, val, adv).
    
    Stratégie de conversion:
    - Les poids conv restent identiques
    - fc1.weight/bias est copié vers fc_val.weight/bias
    - fc2.weight/bias est copié vers fc_adv.weight/bias
    - val.weight/bias et adv.weight/bias sont initialisés aléatoirement (Xavier)
    """
    new_state_dict = {}
    
    # Copier les couches convolutionnelles (identiques)
    for key in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 
                'conv3.weight', 'conv3.bias']:
        if key in old_state_dict:
            new_state_dict[key] = old_state_dict[key]
            print(f"✓ Copié: {key} → {key}")
    
    # Stratégie: L'ancien modèle avait fc1 (3136→512) puis fc2 (512→4)
    # Le nouveau modèle a fc_val ET fc_adv (3136→512) puis val (512→1) et adv (512→n_actions)
    # On va dupliquer fc1 pour fc_val ET fc_adv (comme initialisation raisonnable)
    
    if 'fc1.weight' in old_state_dict:
        # Utiliser fc1 pour fc_val
        new_state_dict['fc_val.weight'] = old_state_dict['fc1.weight'].clone()
        new_state_dict['fc_val.bias'] = old_state_dict['fc1.bias'].clone()
        print(f"✓ Converti: fc1 → fc_val (shape: {old_state_dict['fc1.weight'].shape})")
        
        # Dupliquer fc1 pour fc_adv (avec petite perturbation pour diversité)
        fc_adv_weight = old_state_dict['fc1.weight'].clone()
        fc_adv_weight += torch.randn_like(fc_adv_weight) * 0.01  # Petite perturbation
        new_state_dict['fc_adv.weight'] = fc_adv_weight
        new_state_dict['fc_adv.bias'] = old_state_dict['fc1.bias'].clone()
        print(f"✓ Converti: fc1 → fc_adv (avec perturbation légère)")
    
    # Initialiser les têtes val et adv
    # L'ancien fc2 (512→4) peut servir de base pour adv
    if 'fc2.weight' in old_state_dict:
        # Utiliser fc2 comme base pour adv
        new_state_dict['adv.weight'] = old_state_dict['fc2.weight'].clone()
        new_state_dict['adv.bias'] = old_state_dict['fc2.bias'].clone()
        print(f"✓ Converti: fc2 → adv (shape: {old_state_dict['fc2.weight'].shape})")
    else:
        adv_weight = torch.empty(n_actions, 512)
        torch.nn.init.xavier_uniform_(adv_weight)
        adv_bias = torch.zeros(n_actions)
        new_state_dict['adv.weight'] = adv_weight
        new_state_dict['adv.bias'] = adv_bias
        print(f"✓ Initialisé: adv.weight {adv_weight.shape}, adv.bias {adv_bias.shape}")
    
    # Initialiser val (nouveau dans DuelingDQN)
    val_weight = torch.empty(1, 512)
    torch.nn.init.xavier_uniform_(val_weight)
    val_bias = torch.zeros(1)
    new_state_dict['val.weight'] = val_weight
    new_state_dict['val.bias'] = val_bias
    print(f"✓ Initialisé: val.weight {val_weight.shape}, val.bias {val_bias.shape}")
    
    return new_state_dict


def validate_conversion(new_state_dict, n_actions=4):
    """Vérifie que toutes les clés attendues sont présentes avec les bonnes dimensions."""
    expected_keys = {
        'conv1.weight': (32, 4, 8, 8),
        'conv1.bias': (32,),
        'conv2.weight': (64, 32, 4, 4),
        'conv2.bias': (64,),
        'conv3.weight': (64, 64, 3, 3),
        'conv3.bias': (64,),
        'fc_val.weight': (512, 7*7*64),
        'fc_val.bias': (512,),
        'fc_adv.weight': (512, 7*7*64),
        'fc_adv.bias': (512,),
        'val.weight': (1, 512),
        'val.bias': (1,),
        'adv.weight': (n_actions, 512),
        'adv.bias': (n_actions,),
    }
    
    print("\n🔍 Validation du checkpoint converti:")
    all_valid = True
    for key, expected_shape in expected_keys.items():
        if key not in new_state_dict:
            print(f"  ✗ Clé manquante: {key}")
            all_valid = False
        else:
            actual_shape = tuple(new_state_dict[key].shape)
            if actual_shape == expected_shape:
                print(f"  ✓ {key}: {actual_shape}")
            else:
                print(f"  ✗ {key}: attendu {expected_shape}, obtenu {actual_shape}")
                all_valid = False
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Convertir un ancien checkpoint vers DuelingDQN")
    parser.add_argument('--input', type=str, required=True, help='Chemin du checkpoint à convertir')
    parser.add_argument('--output', type=str, required=True, help='Chemin du checkpoint converti')
    parser.add_argument('--n-actions', type=int, default=4, help='Nombre d\'actions')
    args = parser.parse_args()
    
    print(f"📂 Chargement du checkpoint: {args.input}")
    old_state_dict = torch.load(args.input, map_location='cpu')
    
    print(f"\n🔄 Conversion vers DuelingDQN...")
    new_state_dict = convert_old_to_dueling(old_state_dict, n_actions=args.n_actions)
    
    if validate_conversion(new_state_dict, n_actions=args.n_actions):
        print(f"\n💾 Sauvegarde du checkpoint converti: {args.output}")
        torch.save(new_state_dict, args.output)
        print(f"✅ Conversion terminée avec succès!")
        
        # Test de chargement
        print(f"\n🧪 Test de chargement dans le modèle DuelingDQN...")
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.agent_ddqn import DDQNAgent
        
        agent = DDQNAgent(n_actions=args.n_actions)
        agent.online.load_state_dict(new_state_dict, strict=True)
        print("✅ Le checkpoint converti se charge correctement (strict=True)!")
    else:
        print("\n❌ La validation a échoué. Checkpoint non sauvegardé.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
