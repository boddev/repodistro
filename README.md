# repodistro

 repo_name="gcTemplateCopy"
    file_path = "ApiGraphActivator/BrewerySchema.cs"
    file_content = '''using Microsoft.Graph.Models.ExternalConnectors;

public static class BrewerySchema
{
    public static Schema Schema
    {
        get
        {
            return new Schema
            {
                BaseType = "microsoft.graph.externalItem",
                Properties = new()
                {
                    new Property
                    {
                        Name = "id",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "name",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "brewery_type",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "address_1",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "address_2",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "address_3",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "city",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "state_province",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "postal_code",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "country",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "longitude",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "latitude",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "phone",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "website_url",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "state",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    },
                    new Property
                    {
                        Name = "street",
                        Type = PropertyType.String,
                        IsQueryable = true,
                        IsSearchable = true,
                        IsRetrievable = true,
                        Labels = new string[] {}
                    }
                }
            };
        }
    }
}
'''