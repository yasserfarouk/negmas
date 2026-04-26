package genius.core.repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.repository.boa.BoaPartyRepItem;

/**
 * Contains a list of {@link BoaPartyRepItem}.
 *
 */
@XmlRootElement
public class BoaPartiesList {
	@XmlElement
	private List<BoaPartyRepItem> boaparties = new ArrayList<>();

	public void add(BoaPartyRepItem party) {
		boaparties.add(party);

	}

	public void remove(BoaPartyRepItem party) {
		boaparties.remove(party);
	}

	public List<BoaPartyRepItem> getList() {
		return Collections.unmodifiableList(boaparties);
	}

	public void addAll(List<BoaPartyRepItem> values) {
		boaparties.addAll(values);
	}

}
