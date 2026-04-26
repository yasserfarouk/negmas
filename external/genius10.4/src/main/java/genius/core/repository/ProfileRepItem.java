package genius.core.repository;

import java.net.URL;

import javax.xml.bind.Unmarshaller;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlTransient;

import genius.core.Domain;
import genius.core.exceptions.Warning;
import genius.core.session.RepositoryException;
import genius.core.utility.UtilitySpace;

/**
 * ProfileRepItem is a profile, as an item to put in the registry. The profile
 * is not contained here, it's just a (assumed unique) filename. immutable.
 * 
 * @author W.Pasman added code to unmarshall this if not part of a
 *         domainrepository.xml file. We then search for this profile in the
 *         existing domain repository. see
 *         {@link #afterUnmarshal(Unmarshaller, Object)}.
 *
 */
@XmlRootElement
public class ProfileRepItem implements RepItem {
	private static final long serialVersionUID = -5071749178482314158L;
	@XmlAttribute
	private URL url;

	@XmlTransient
	private DomainRepItem domain;

	/** needed by XML serializer */
	@SuppressWarnings("unused")
	protected ProfileRepItem() {
		try {
			url = new URL("file:xml-failed-to-set-url");
		} catch (Exception e) {
			new Warning("failed to set filename default value" + e);
		}
	}

	public ProfileRepItem(URL file, DomainRepItem dom) {
		url = file;
		domain = dom;
	}

	public URL getURL() {
		return url;
	}

	public DomainRepItem getDomain() {
		return domain;
	}

	@Override
	public String toString() {
		return getURL().getFile();
	}

	/**
	 * @return a full name but without any special characters like "/" and ":".
	 */
	public String getFullName() {
		return url.toString().replaceAll("/", ".").replaceAll(":", ".");
	}

	/**
	 * See {@link Unmarshaller}.
	 * 
	 * @param u
	 * @param parent
	 */
	public void afterUnmarshal(Unmarshaller u, Object parent) {
		if (parent instanceof DomainRepItem) {
			domain = (DomainRepItem) parent;
		} else {
			domain = searchDomain();
		}
	}

	/**
	 * Try to find this profile in the domain repository. This is needed if this
	 * profilerepitem is not part of a domainrepository.xml file.
	 * 
	 * @return DomainRepItem. Returns null if this profile is part of an
	 *         existing domain in the repository
	 */
	private DomainRepItem searchDomain() {
		// this ProfileRepItem is not in a domain repository. Try to resolve
		// domain using the repository
		try {
			for (RepItem item : RepositoryFactory.get_domain_repos().getItems()) {
				DomainRepItem repitem = (DomainRepItem) item;
				if (repitem.getProfiles().contains(this)) {
					return repitem;
				}
			}
		} catch (RepositoryException e) {
			e.printStackTrace();
		}
		System.err.println("The profile " + this + " is not in the domain repository, failed to unmarshall");
		throw new NullPointerException("no domain found");
	}

	@Override
	public int hashCode() {
		int hash = 7;
		hash = 97 * hash + (this.url != null ? this.url.hashCode() : 0);
		return hash;
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof ProfileRepItem)) {
			return false;
		}
		return url.equals(((ProfileRepItem) o).getURL());
	}

	public String getName() {
		String name = url.getFile();
		if (name.contains("/") && name.contains(".")) {
			name = name.substring(name.lastIndexOf("/") + 1, name.lastIndexOf("."));
		}
		return name;
	}

	/**
	 *
	 * @return a new UtilitySpace referred to by this ProfileRepItem. If
	 *         {@link ProfileRepItem#getDomain()} returns new instead of an
	 *         actual domain, this method also returns null.
	 * @throws RepositoryException
	 */
	public UtilitySpace create() throws RepositoryException {
		Domain domain;
		try {
			domain = RepositoryFactory.get_domain_repos().getDomain(getDomain());

			return RepositoryFactory.get_domain_repos().getUtilitySpace(domain, this);
		} catch (Exception e) {
			throw new RepositoryException("File not found for " + this, e);
		}
	}

}
