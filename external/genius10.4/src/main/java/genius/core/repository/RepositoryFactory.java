package genius.core.repository;

import java.io.File;
import java.io.IOException;

import javax.xml.bind.JAXBException;

import genius.AgentsInstaller;
import genius.ProtocolsInstaller;
import genius.core.repository.boa.BoaRepository;
import genius.core.session.RepositoryException;
import genius.domains.DomainInstaller;

/**
 * Factory for Repositories.
 */
public class RepositoryFactory {

	private static final String FILENAME = "domainrepository.xml";
	// ASSUMPTION there is only one domain repository

	// cache for domain repo.
	private static Repository<DomainRepItem> domainRepos = null;

	// cache for BoaPartyRepository
	private static BoaPartyRepository boapartyrepo = null;

	private static BoaRepository boarepo = null;

	private static boolean installedProtocols = false;

	public static DomainRepItem getDomainByName(String name) throws Exception {
		Repository<DomainRepItem> domRep = get_domain_repos();
		DomainRepItem domainRepItem = null;
		for (RepItem tmp : domRep.getItems()) {
			if (((DomainRepItem) tmp).getURL().toString().equals(name)) {
				domainRepItem = (DomainRepItem) tmp;
				break;
			}
		}
		return domainRepItem;
	}

	public static Repository<DomainRepItem> get_domain_repos(String filename, String sourceFolder)
			throws RepositoryException {
		if (domainRepos == null) {
			try {
				DomainInstaller.run();
				domainRepos = Repository.fromFile(filename);
			} catch (IOException e) {
				throw new RepositoryException("Failed to load domains", e);
			}
		}
		return domainRepos;
	}

	public static Repository<DomainRepItem> get_domain_repos() throws RepositoryException {
		return get_domain_repos(FILENAME, "");

	}

	private static void installProtocols() {
		if (!installedProtocols) {
			try {
				ProtocolsInstaller.run();
				installedProtocols = true;
			} catch (IOException e) {
				throw new RepositoryException("Failed to install protocols", e);
			}
		}
	}

	public static Repository<ProtocolRepItem> getProtocolRepository() throws RepositoryException {
		installProtocols();
		return Repository.fromFile("protocolrepository.xml");
	}

	public static Repository<MultiPartyProtocolRepItem> getMultiPartyProtocolRepository() throws RepositoryException {
		installProtocols();
		return Repository.fromFile("multipartyprotocolrepository.xml");

	}

	public static Repository<AgentRepItem> get_agent_repository() throws RepositoryException {
		return Repository.fromFile("agentrepository.xml");

	}

	public static Repository<PartyRepItem> get_party_repository() throws RepositoryException {
		return Repository.fromFile("partyrepository.xml");
	}

	public static BoaPartyRepository getBoaPartyRepository() throws RepositoryException {
		if (boapartyrepo == null) {
			try {
				AgentsInstaller.run();
				// FIXME use Repository.fromFile
				boapartyrepo = new BoaPartyRepository();
			} catch (JAXBException | IOException e) {
				throw new RepositoryException("failed to load BoaPartyRepo", e);
			}
		}
		return boapartyrepo;
	}

	public static BoaRepository getBoaRepository() {
		if (boarepo == null) {
			try {
				AgentsInstaller.run();
				boarepo = BoaRepository.loadRepository(new File("boarepository.xml"));
			} catch (JAXBException | IOException e) {
				throw new RepositoryException("failed to load BoaRepo", e);
			}
		}
		return boarepo;
	}

}
