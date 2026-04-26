package genius.core.xml;

import java.util.ArrayList;
import java.util.List;

/**
 * Orders the attributes
 */
public class OrderedSimpleElement extends SimpleElement
{
	int index = 0;
	List<String> attributeKeys;
	
	public OrderedSimpleElement(String tagName)
	{
		super(tagName);
		attributeKeys = new ArrayList<String>();
	}
	
	@Override
	public void setAttribute(String name, String value) 
	{
		attributes.put(name, value);
		if (!attributeKeys.contains(name))
			attributeKeys.add(name);
	}
	
	@Override
	public String toString() {
    	String lResult="";
    	lResult +="<" + tagName;
        //save all attributes
        for(int i=0;i<attributeKeys.size();i++) {
        	String lAttrName = attributeKeys.get(i);
        	String lAttrValue="";
        	if (attributes.entrySet().toArray()[i]!=null)
        		lAttrValue= (attributes.get(lAttrName));
        	
        	lResult +=" "+lAttrName+"=\"" +lAttrValue+"\"";
        }
        lResult +="> \n";
        //save all children
        for(int i=0;i<childElements.size();i++) {
        	SimpleElement lSE = (SimpleElement)getChildElements()[i];
        	lResult += lSE.toString();
        }
        if(text!=null) {
        	lResult += text+" \n";}
        lResult +="</" + tagName+"> \n";
    	
    	return lResult;
    }

}
