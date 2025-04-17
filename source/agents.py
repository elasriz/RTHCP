

def agent_factory(env, observer, agent_class: str, delay:int, **params: dict):
    
    if agent_class == "RTHCP":
        from source.RTHCP import RTHCP
        return RTHCP(env, observer, delay, **params)

    
    elif agent_class == "RandomPolicy":
        from source.random_policy import RandomPolicy
        return RandomPolicy(env, **params)
    
