import asyncio
from my_env import BrokenPipelineEnv, BrokenPipelineAction

async def main():
    print("Testing my_env.py...")
    env = BrokenPipelineEnv()
    
    print("\n--- Resetting ---")
    res = await env.reset()
    print("Reset Result:", res)
    
    print("\n--- Stepping (diagnose) ---")
    res = await env.step(BrokenPipelineAction(message="diagnose"))
    print("Step Result:", res)

    await env.close()
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main())
