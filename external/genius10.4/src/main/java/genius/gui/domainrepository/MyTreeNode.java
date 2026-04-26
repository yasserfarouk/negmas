package genius.gui.domainrepository;

import javax.swing.tree.DefaultMutableTreeNode;

import genius.core.exceptions.Warning;
import genius.core.repository.DomainRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.RepItem;

public class MyTreeNode extends DefaultMutableTreeNode {

	private static final long serialVersionUID = -4929729243877782033L;
	RepItem repository_item;
	
	public MyTreeNode(RepItem item)
	{
		super(item);
		repository_item=item;
	}
	
	public String toString() {
		if (repository_item==null) return "";
		if (repository_item instanceof DomainRepItem)
			return shortfilename(((DomainRepItem)repository_item).getURL().getFile());
		if (repository_item instanceof ProfileRepItem)
			return shortfilename( ((ProfileRepItem)repository_item).getURL().getFile());
		new Warning("encountered item "+repository_item+" of type "+repository_item.getClass());
		return "ERR";
	}
	/** returns only the filename given a full path with separating '/' */
	public String shortfilename(String filename) {
		int lastslash=filename.lastIndexOf('/');
		if (lastslash==-1) return filename;
		return filename.substring(lastslash+1); 
	}
	
	public RepItem getRepositoryItem() { return repository_item; }
}