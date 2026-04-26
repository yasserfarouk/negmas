package genius.core.repository;

import java.util.ArrayList;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;

public class RepositoryItemType {
    
    // produce a wrapper XML element around this collection
    @XmlElementWrapper(name = "agentRepItems")
    @XmlElement(name = "agentRepItem")
    private ArrayList<AgentRepItem> agentRepItem;
    
    
    // produce a wrapper XML element around this collection
    @XmlElementWrapper(name = "partyRepItems")
    @XmlElement(name = "partyRepItem")
    private ArrayList<PartyRepItem> partyRepItem;
    
    // produce a wrapper XML element around this collection
    @XmlElementWrapper(name = "domainRepItem")
    @XmlElement(name = "domainRepItem")
    private ArrayList<DomainRepItem> domainRepItem;
    
    
    // produce a wrapper XML element around this collection
    @XmlElementWrapper(name = "protocolRepItem")
    @XmlElement(name = "protocolRepItem")
    private ArrayList<ProtocolRepItem> protocolRepItem;
    
 // produce a wrapper XML element around this collection
    @XmlElementWrapper(name = "multiPartyProtocolRepItem")
    @XmlElement(name = "multiPartyProtocolRepItem")
    private ArrayList<MultiPartyProtocolRepItem> multiPartyProtocolRepItem;
    
    
    public RepositoryItemType() {}
    
    public RepositoryItemType(ArrayList<AgentRepItem> agentRepItem,ArrayList<PartyRepItem> partyRepItem, ArrayList<DomainRepItem> domainRepItem, ArrayList<ProtocolRepItem> protocolRepItem, ArrayList<MultiPartyProtocolRepItem> multiPartyProtocolRepItem) {
        this.agentRepItem = agentRepItem;
        this.partyRepItem=partyRepItem;
        this.domainRepItem = domainRepItem;
        this.protocolRepItem = protocolRepItem;
        this.multiPartyProtocolRepItem=multiPartyProtocolRepItem;
    }

    public ArrayList<AgentRepItem> getAgentRepItem() {
        return agentRepItem;
    }
    
    public ArrayList<PartyRepItem> getPartyRepItem() {
        return partyRepItem;
    }
    
    public ArrayList<DomainRepItem> getDomainRepItem() {
        return domainRepItem;
    }
    
    public ArrayList<ProtocolRepItem> getProtocolRepItem() {
    	return protocolRepItem;
    }
    
    public ArrayList<MultiPartyProtocolRepItem> getMultiPartyProtocolRepItem() {
    	return multiPartyProtocolRepItem;
    }
}


