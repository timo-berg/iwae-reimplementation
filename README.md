# Overview over the reposity
As outlined in the report, there are three different folders containing our own jax implementation, 
the pytorch reimplementation of that, and a copy of the 
[repository](https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py) 
based on which we assessed the IWAE. 
The jax reimplementation suffers from a dimensional mismatch that we were unable to fix in the given amount 
of time a and only contains a rudimentary training loop.
The reimplementation `/pytorch_reimplementation` does not work as intended yet but we are confident that we
could obtain a functioning implementation in the near future.
