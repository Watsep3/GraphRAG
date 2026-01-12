from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("DIAGNOSTIC NEO4J")
print("="*60)

# Lire les credentials
uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASSWORD')

print(f"\n📋 Configuration:")
print(f"  URI: {uri}")
print(f"  User: {user}")
print(f"  Password: {'*' * len(password) if password else 'NON DÉFINI'}")

# Test de connexion
print(f"\n🔌 Test de connexion...")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Test simple
        result = session.run("RETURN 1 as test")
        result.single()
        
        print("✅ CONNEXION RÉUSSIE!")
        
        # Statistiques
        result = session.run("MATCH (n) RETURN count(n) as count")
        node_count = result.single()['count']
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()['count']
        
        print(f"\n📊 Statistiques:")
        print(f"  Nœuds: {node_count:,}")
        print(f"  Relations: {rel_count:,}")
        
        if node_count == 0:
            print("\n⚠️  LA BASE EST VIDE!")
            print("   Chargez les données avec: python src/load_neo4j_v2.py")
    
    driver.close()
    
except ServiceUnavailable as e:
    print("❌ ERREUR: Neo4j n'est pas accessible")
    print(f"   Détails: {e}")
    print("\n📋 Solutions:")
    print("   1. Vérifiez que Neo4j Desktop est ouvert")
    print("   2. Vérifiez que votre instance est DÉMARRÉE (point vert)")
    print("   3. Vérifiez le port (doit être 7687)")

except AuthError as e:
    print("❌ ERREUR: Authentification échouée")
    print(f"   Détails: {e}")
    print("\n📋 Solutions:")
    print("   1. Vérifiez le mot de passe dans .env")
    print("   2. Réinitialisez le mot de passe dans Neo4j Desktop")

except Exception as e:
    print(f"❌ ERREUR INCONNUE: {e}")
    print(f"   Type: {type(e)}")

print("\n" + "="*60)