# attack
# QSA
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model gcn --surrogate gcn --topk 50 --m 1 --retrain
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model gat --surrogate gcn --topk 50 --m 1
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model decor --surrogate gcn --topk 50 --m 1
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model mid-gcn --surrogate gcn --topk 50 --m 1

python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model gcn --surrogate gcn --topk 50 --m 2 --retrain
python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model gat --surrogate gcn --topk 50 --m 2
python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model decor --surrogate gcn --topk 50 --m 2
python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model mid-gcn --surrogate gcn --topk 50 --m 2

# QSA-AC
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model gcn --surrogate gcn --topk 50 --m 1 --constrain --alpha 1 --retrain
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model gat --surrogate gcn --topk 50 --m 1 --constrain --alpha 1
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model decor --surrogate gcn --topk 50 --m 1 --constrain --alpha 1
python -u attack_method/test_qsa_ac.py --dataset politifact --victim_model mid-gcn --surrogate gcn --topk 50 --m 1 --constrain --alpha 1

python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model gcn --surrogate gcn --topk 50 --m 2 --constrain  --alpha 1 --retrain
python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model gat --surrogate gcn --topk 50 --m 2 --constrain  --alpha 1
python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model decor --surrogate gcn --topk 50 --m 2 --constrain  --alpha 1
python -u attack_method/test_qsa_ac.py --dataset gossipcop --subset subset1 --victim_model mid-gcn --surrogate gcn --topk 50 --m 2 --constrain  --alpha 1