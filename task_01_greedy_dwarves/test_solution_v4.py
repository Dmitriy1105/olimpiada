import random
from solution_v4 import GreedyDwarvesAgent

def test_no_fitting_item_forced_pass():
    a=GreedyDwarvesAgent(2,3,5,10,[11,12,13],[10,20,30],cpu_budget=100)
    assert a.choose()==-1
    a.commit_own_action(-1)
    a.observe([0])
    assert a.round_no==1

def test_random_shared_policy_games():
    for seed in range(100):
        rng=random.Random(seed)
        k=rng.randint(2,10); n=rng.randint(10,100)
        rounds=rng.randint(5,20); capacity=rng.randint(10,50)
        masses=[rng.randint(1,50) for _ in range(n)]
        values=[rng.randint(1,100) for _ in range(n)]
        a=GreedyDwarvesAgent(k,n,rounds,capacity,masses,values,cpu_budget=100)
        hidden=rng.randrange(a.p_count)
        for _ in range(rounds):
            item=a.choose()
            if item<0:
                assert not any(((a.available_mask>>i)&1) and masses[i]<=capacity-a.player_mass[0] for i in range(n))
            else:
                assert 0<=item<n and ((a.available_mask>>item)&1)
                assert a.player_mass[0]+masses[item]<=capacity
            a.commit_own_action(item)
            bot_mask=a.remaining_masks[hidden]
            actions=[]
            for j in range(1,k):
                free=capacity-a.player_mass[j]
                legal=bot_mask&a.capacity_masks[hidden][free] if free>0 else 0
                if legal:
                    rank=(legal&-legal).bit_length()-1
                    bot_item=a.orders[hidden][rank]
                    bot_mask &= ~a.item_bits[hidden][bot_item]
                    actions.append(bot_item+1)
                else:
                    actions.append(0)
            a.observe(actions)
            assert all(mass<=capacity for mass in a.player_mass)

def test_posterior_normalized():
    a=GreedyDwarvesAgent(3,8,5,10,[7,7,1,5,9,8,7,5],[62,46,75,28,65,18,37,18],cpu_budget=100)
    a.commit_own_action(0)
    old=a.posterior[:]
    a.observe([2,4])
    assert a.posterior!=old
    assert abs(sum(a.posterior)-1.0)<1e-9
    assert all(p>0 for p in a.posterior)

if __name__=="__main__":
    test_no_fitting_item_forced_pass()
    test_random_shared_policy_games()
    test_posterior_normalized()
    print("V4 tests: PASS (100 randomized games + edge cases)")
