package parties.AlternatingMultipleOffers;

import genius.core.protocol.AlternatingMajorityConsensusProtocol;
import genius.core.protocol.DefaultMultilateralProtocol;

public class RandomAmopPartyMajority extends RandomAmopParty {

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return AlternatingMajorityConsensusProtocol.class;
	}
}
