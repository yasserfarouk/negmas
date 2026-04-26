package genius.core.repository.boa;

import java.util.Map;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOA;
import genius.core.boaframework.BoaParty;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.exceptions.InstantiateException;
import genius.core.parties.NegotiationParty;
import genius.core.repository.ParticipantRepItem;

/**
 * This repository item contains all info about a {@link NegotiationParty} that
 * can be loaded by construction with BOAcomponents. immutable.
 *
 */
@SuppressWarnings("serial")
@XmlRootElement(name = "boaparty")
public class BoaPartyRepItem extends ParticipantRepItem {
	@XmlAttribute
	private String partyName = "unknown";

	/** Offering strategy of the specified agent */
	@XmlElement
	private BoaWithSettingsRepItem<OfferingStrategy> biddingStrategy;

	/** Acceptance strategy of the specified agent */
	@XmlElement
	private BoaWithSettingsRepItem<AcceptanceStrategy> acceptanceStrategy;

	/** Opponent model of the specified agent */
	@XmlElement
	private BoaWithSettingsRepItem<OpponentModel> opponentModel;

	/** Opponent model strategy of the specified agent */
	@XmlElement
	private BoaWithSettingsRepItem<OMStrategy> omStrategy;

	// for serializer
	@SuppressWarnings("unused")
	private BoaPartyRepItem() {
	}

	/**
	 * Create default Boa party with given name.
	 * 
	 * @param partyName
	 *            name for the default party
	 * @throws InstantiateException
	 *             if problem with repo
	 */
	public BoaPartyRepItem(String partyName) throws InstantiateException {
		this.partyName = partyName;
		biddingStrategy = new BoaWithSettingsRepItem<>(BoaType.BIDDINGSTRATEGY);
		acceptanceStrategy = new BoaWithSettingsRepItem<>(
				BoaType.ACCEPTANCESTRATEGY);
		opponentModel = new BoaWithSettingsRepItem<>(BoaType.OPPONENTMODEL);
		omStrategy = new BoaWithSettingsRepItem<>(BoaType.OMSTRATEGY);
	}

	public BoaPartyRepItem(String partyName,
			BoaWithSettingsRepItem<OfferingStrategy> boa1,
			BoaWithSettingsRepItem<AcceptanceStrategy> boa2,
			BoaWithSettingsRepItem<OpponentModel> boa3,
			BoaWithSettingsRepItem<OMStrategy> boa4) {
		if (boa1 == null || boa2 == null || boa3 == null || boa4 == null) {
			throw new NullPointerException(
					"boa arguments must all be not null");
		}
		this.partyName = partyName;
		this.biddingStrategy = boa1;
		this.acceptanceStrategy = boa2;
		this.opponentModel = boa3;
		this.omStrategy = boa4;
	}

	@Override
	public String getName() {
		return partyName;
	}

	public BoaWithSettingsRepItem<OfferingStrategy> getOfferingStrategy() {
		return biddingStrategy;
	}

	public BoaWithSettingsRepItem<AcceptanceStrategy> getAcceptanceStrategy() {
		return acceptanceStrategy;
	}

	public BoaWithSettingsRepItem<OpponentModel> getOpponentModel() {
		return opponentModel;
	}

	public BoaWithSettingsRepItem<OMStrategy> getOmStrategy() {
		return omStrategy;
	}

	public BoaWithSettingsRepItem<? extends BOA> getStrategy(
			BoaType strategyType) {
		switch (strategyType) {
		case BIDDINGSTRATEGY:
			return biddingStrategy;
		case ACCEPTANCESTRATEGY:
			return acceptanceStrategy;
		case OPPONENTMODEL:
			return opponentModel;
		case OMSTRATEGY:
			return omStrategy;
		}
		throw new IllegalArgumentException(
				"There is no strategy for " + strategyType);
	}

	@Override
	public int hashCode() {
		return 0;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		BoaPartyRepItem other = (BoaPartyRepItem) obj;
		if (acceptanceStrategy == null) {
			if (other.acceptanceStrategy != null)
				return false;
		} else if (!acceptanceStrategy.equals(other.acceptanceStrategy))
			return false;
		if (biddingStrategy == null) {
			if (other.biddingStrategy != null)
				return false;
		} else if (!biddingStrategy.equals(other.biddingStrategy))
			return false;
		if (omStrategy == null) {
			if (other.omStrategy != null)
				return false;
		} else if (!omStrategy.equals(other.omStrategy))
			return false;
		if (opponentModel == null) {
			if (other.opponentModel != null)
				return false;
		} else if (!opponentModel.equals(other.opponentModel))
			return false;
		if (partyName == null) {
			if (other.partyName != null)
				return false;
		} else if (!partyName.equals(other.partyName))
			return false;
		return true;
	}

	@Override
	public NegotiationParty load() throws InstantiateException {
		OfferingStrategy os = biddingStrategy.getBoa().getInstance();
		Map<String, Double> osparams = biddingStrategy.getParameters().asMap();
		AcceptanceStrategy as = acceptanceStrategy.getBoa().getInstance();
		Map<String, Double> asparams = acceptanceStrategy.getParameters()
				.asMap();
		OpponentModel om = opponentModel.getBoa().getInstance();
		Map<String, Double> omparams = opponentModel.getParameters().asMap();
		OMStrategy oms = omStrategy.getBoa().getInstance();
		Map<String, Double> omsparams = omStrategy.getParameters().asMap();
		BoaParty boa = new BoaParty() {
			@Override
			public String getDescription() {
				return "BOA(" + as.getName() + "," + os.getName() + ","
						+ om.getName() + "," + oms.getName() + ")";
			}
		};
		boa.configure(as, asparams, os, osparams, om, omparams, oms, omsparams);
		return boa;
	}

	@Override
	public String getUniqueName() {
		return "boa-" + biddingStrategy.getUniqueName() + "-"
				+ acceptanceStrategy.getUniqueName() + "-"
				+ opponentModel.getUniqueName() + "-"
				+ omStrategy.getUniqueName();
	}

	@Override
	public String toString() {
		return "BoaPartyRepItem[" + partyName + "," + biddingStrategy + ","
				+ acceptanceStrategy + "," + opponentModel + "," + omStrategy
				+ "]";
	}

	@Override
	public String getClassDescriptor() {
		return toString();
	}

}
