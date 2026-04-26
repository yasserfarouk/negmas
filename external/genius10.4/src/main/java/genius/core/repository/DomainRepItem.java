package genius.core.repository;

import java.net.URL;
import java.util.ArrayList;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.exceptions.Warning;

/**
 * A DomainRepItem is a domain reference that can be put in the domain
 * repository. It contains only a unique reference to an xml file with the
 * domain description.
 * 
 * @author wouter
 */
@XmlRootElement
public class DomainRepItem implements RepItem {
	private static final long serialVersionUID = 6672725212678925392L;
	@XmlAttribute
	private URL url;
	@XmlElement(name = "profile")
	private ArrayList<ProfileRepItem> profiles = new ArrayList<ProfileRepItem>(); // default

	/** for serialization support */
	@SuppressWarnings("unused")
	private DomainRepItem() {
		try {
			url = new URL("file:unknownfilename");
		} catch (Exception e) {
			new Warning("default url failed!?", e);
		}
	}

	public DomainRepItem(URL newurl) {
		url = newurl;
	}

	public URL getURL() {
		return url;
	}

	public ArrayList<ProfileRepItem> getProfiles() {
		return profiles;
	}

	@Override
	public String toString() {
		return getName();
	}

	public String getFullName() {
		return "DomainRepItem[" + url + "," + profiles + "]";
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof DomainRepItem)) {
			return false;
		}
		return url.equals(((DomainRepItem) o).getURL());
	}

	@Override
	public int hashCode() {
		int hash = 7;
		hash = 89 * hash + (this.url != null ? this.url.hashCode() : 0);
		return hash;
	}

	public String getName() {
		String name = url.getFile();
		int dotindex = name.lastIndexOf('.'), slashindex = name
				.lastIndexOf('/');

		if (slashindex < 0)
			slashindex = 0;
		if (dotindex < 0)
			dotindex = name.length() - 1;
		name = name.substring(slashindex + 1, dotindex);

		return name;
	}

}
