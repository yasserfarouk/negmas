package genius.core.repository;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.adapters.XmlJavaTypeAdapter;
import javax.xml.namespace.QName;

import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.exceptions.Warning;
import genius.core.listener.DefaultListenable;
import genius.core.repository.boa.BoaPartyRepItem;
import genius.core.session.RepositoryException;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.ConstraintUtilitySpace;
import genius.core.utility.NonlinearUtilitySpace;
import genius.core.utility.UTILITYSPACETYPE;
import genius.core.utility.UncertainAdditiveUtilitySpace;

/**
 * Repository contains a set of known files This can be agent files or
 * domain+profile files.
 * 
 * @author Dmytro Tykhonov, W.Pasman
 * 
 * @param T
 *            the repository element type
 */
@XmlRootElement
public class Repository<T extends RepItem>
		extends DefaultListenable<BoaPartyRepItem> {

	@XmlJavaTypeAdapter(RepositoryItemTypeAdapter.class)
	ArrayList<T> items = new ArrayList<T>(); // the items in the domain

	@XmlAttribute
	String fileName; // the filename of this repository.

	String sourceFolder = null;

	// only for unmarshaller.
	@SuppressWarnings("unused")
	private Repository() {
	}

	public String getFilename() {
		return fileName;
	}

	/**
	 * 
	 * @param <T1>
	 *            type of the {@link RepItem} elements in this repo.
	 * @param fileName
	 *            the file containing {@link Repository}
	 * @return repository loaded from given file.
	 */
	public static <T1 extends RepItem> Repository<T1> fromFile(
			String fileName) {
		try {
			JAXBContext jaxbContext = JAXBContext.newInstance(Repository.class,
					PartyRepItem.class, ProfileRepItem.class,
					MultiPartyProtocolRepItem.class, DomainRepItem.class,
					AgentRepItem.class);
			Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();
			unmarshaller.setEventHandler(
					new javax.xml.bind.helpers.DefaultValidationEventHandler());
			Repository<T1> rep = (Repository<T1>) (unmarshaller
					.unmarshal(new File(fileName)));
			rep.fileName = fileName;
			return rep;
		} catch (JAXBException e) {
			throw new RepositoryException("Failed to unmarshall " + fileName,
					e);
		}
	}

	/**
	 * Save this to the file.
	 */
	public void save() {
		try {
			JAXBContext jaxbContext = JAXBContext.newInstance(Repository.class,
					ProfileRepItem.class, DomainRepItem.class,
					AgentRepItem.class, PartyRepItem.class,
					ProtocolRepItem.class, MultiPartyProtocolRepItem.class);
			Marshaller marshaller = jaxbContext.createMarshaller();
			marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT,
					new Boolean(true));

			marshaller.marshal(new JAXBElement(new QName("repository"),
					Repository.class, this), new File(fileName));
		} catch (Exception e) {
			new Warning("xml save failed: " + e);
		}

	}

	/**
	 * @return available agents
	 */
	public ArrayList<T> getItems() {
		return items;
	}

	/** @return AgentRepItem of given className, or null if none exists */
	public AgentRepItem getAgentOfClass(String className) {
		for (RepItem it : items) {
			if (it instanceof AgentRepItem)
				if (((AgentRepItem) it).getClassPath().equals(className))
					return (AgentRepItem) it;
		}
		return null;
	}

	/**
	 * @param className
	 *            the full class name to load.
	 * @return AgentRepItem of given className, or null if none exists
	 */
	public PartyRepItem getPartyOfClass(String className) {
		for (RepItem it : items) {
			if (it instanceof PartyRepItem)
				if (((PartyRepItem) it).getClassPath().equals(className))
					return (PartyRepItem) it;
		}
		return null;
	}

	public boolean removeProfileRepItem(ProfileRepItem item) {
		for (int i = 0; i < items.size(); i++) {
			System.out.println(items.get(i).getName());
			DomainRepItem drp = (DomainRepItem) items.get(i);
			for (int a = 0; a < drp.getProfiles().size(); a++) {
				ProfileRepItem pri = drp.getProfiles().get(a);
				if (pri.getName().equals(item.getName())) {
					drp.getProfiles().remove(a);
					return true;
				}
			}
		}
		return false;
	}

	@Override
	public String toString() {
		String ret = "{";
		for (RepItem i : items) {
			ret = ret + i + ",";
		}
		ret = ret + "}";
		return ret;

	}

	public Domain getDomain(DomainRepItem domainRepItem) throws IOException {
		String file = domainRepItem.getURL().getFile();
		return getDomain(file);
	}

	public Domain getDomain(String file) throws IOException {
		Domain domain = null;
		if ((sourceFolder != null) && (!sourceFolder.equals("")))
			domain = new DomainImpl(sourceFolder + "/" + file);
		else
			domain = new DomainImpl(file);
		return domain;
	}

	public AbstractUtilitySpace getUtilitySpace(Domain domain,
			ProfileRepItem profile) {
		String file = profile.getURL().getFile();
		return getUtilitySpace(domain, file);
	}

	public AbstractUtilitySpace getUtilitySpace(Domain domain, String file) {
		AbstractUtilitySpace us = null;
		String fullfile;
		if ((sourceFolder != null) && (!sourceFolder.equals(""))) {
			fullfile = sourceFolder + "/" + file;
		} else {
			fullfile = file;
		}

		try {
			switch (UTILITYSPACETYPE.getUtilitySpaceType(file)) {
			case NONLINEAR:
				us = new NonlinearUtilitySpace(domain, fullfile);
				break;
			case CONSTRAINT:
				us = new ConstraintUtilitySpace(domain, fullfile);
				break;
			case UNCERTAIN:
				us = new UncertainAdditiveUtilitySpace(domain, fullfile);
				break;
			case LINEAR:
			default:
				us = new AdditiveUtilitySpace(domain, fullfile);
				break;
			}
		} catch (Exception e) {
			System.out.println("Failed to load space:" + file);
			e.printStackTrace();
		}
		return us;
	}

	public boolean existUtilitySpace(Domain domain, ProfileRepItem profile) {

		try {
			File file;
			if ((sourceFolder != null) && (!sourceFolder.equals("")))
				file = new File(
						sourceFolder + "/" + profile.getURL().getFile());
			else
				file = new File(profile.getURL().getFile());
			return file.exists();
		} catch (Exception e) {
			System.out.println(
					"Failed to load space:" + profile.getURL().getFile());
			e.printStackTrace();
		}
		return false;
	}

	public RepItem getItemByName(String name) {
		for (RepItem ri : items) {
			// get protocol name
			if (ri.getName().equals(name))
				return ri;
		}
		return null;
	}
}