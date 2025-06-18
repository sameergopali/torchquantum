#!/usr/bin/env python3
"""Interactive setup script for IBM Quantum Runtime credentials."""

import sys
import os

def main():
    """Interactive setup for IBM Quantum credentials."""
    
    print("🌐 IBM Quantum Runtime Setup")
    print("=" * 40)
    
    # Check if qiskit-ibm-runtime is installed
    try:
        import qiskit_ibm_runtime
        print(f"✅ qiskit-ibm-runtime installed: {qiskit_ibm_runtime.__version__}")
    except ImportError:
        print("❌ qiskit-ibm-runtime not found!")
        print("Install it with: pip install qiskit-ibm-runtime")
        return False
    
    print("\n📋 Setup Instructions:")
    print("1. Go to: https://quantum-computing.ibm.com/")
    print("2. Create an account or log in")
    print("3. Click on your profile → Account → API token")
    print("4. Copy your API token")
    
    # Get token from user
    print("\n🔑 Enter your IBM Quantum API token:")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return False
    
    # Ask for channel and instance
    print("\n🏛️ Select channel:")
    print("1. ibm_quantum (Free/Premium IBM Quantum Network)")
    print("2. ibm_cloud (IBM Cloud)")
    
    channel_choice = input("Choice (1/2): ").strip()
    
    if channel_choice == "1":
        channel = "ibm_quantum"
        print("\n🏢 Enter instance (format: hub/group/project):")
        print("Default for open access: ibm-q/open/main")
        instance = input("Instance (press Enter for default): ").strip()
        if not instance:
            instance = "ibm-q/open/main"
    elif channel_choice == "2":
        channel = "ibm_cloud"
        instance = None
    else:
        print("❌ Invalid choice. Using default: ibm_quantum")
        channel = "ibm_quantum"
        instance = "ibm-q/open/main"
    
    # Save credentials
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        print(f"\n💾 Saving credentials...")
        print(f"   Channel: {channel}")
        if instance:
            print(f"   Instance: {instance}")
        
        QiskitRuntimeService.save_account(
            token=token,
            channel=channel,
            instance=instance,
            overwrite=True
        )
        
        print("✅ Credentials saved successfully!")
        
        # Test connection
        print("\n🧪 Testing connection...")
        service = QiskitRuntimeService()
        backends = service.backends()
        
        print(f"✅ Connected! Found {len(backends)} available backends")
        
        # Show a few backends
        if backends:
            print("\n📋 Sample backends:")
            for i, backend in enumerate(backends[:5]):
                type_icon = "🖥️" if backend.simulator else "🔬"
                print(f"   {type_icon} {backend.name}: {backend.num_qubits} qubits")
            if len(backends) > 5:
                print(f"   ... and {len(backends) - 5} more")
        
        print("\n🎉 Setup complete! You can now use IBM Quantum hardware.")
        print("\nNext steps:")
        print("1. Run: python test_hardware_connection.py")
        print("2. Try the advanced examples with real hardware")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\nCommon issues:")
        print("• Invalid token - check it's copied correctly")
        print("• Network connectivity issues")
        print("• Account permissions")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to run quantum circuits on IBM hardware!")
    else:
        print("\n⚠️ Setup incomplete. Please try again.")
    
    sys.exit(0 if success else 1) 