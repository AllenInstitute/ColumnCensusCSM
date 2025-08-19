import click
import dotenv
from caveclient import CAVEclient

process_dir = "synapse_files"


def data_filename(data, client):
    return f"{data}_df_v{client.materialize.version}.feather"


@click.command()
@click.option("-c", "--config", required=False, default=None)
@click.option("-t", "--cell_table", envvar="CELL_TYPE_TABLE", type=str)
@click.option("-s", "--soma_table", envvar="SOMA_TABLE", type=str)
@click.option("-v", "--version", required=False, default=None, type=int)
@click.option("-d", "--datastack", envvar="DATASTACK", type=str)
def cli(config, cell_table, soma_table, version, datastack):
    if config is not None:
        vals = dotenv.dotenv_values(config)
        datastack = vals.get("DATASTACK", datastack)
        cell_table = vals.get("CELL_TYPE_TABLE", cell_table)
        soma_table = vals.get("SOMA_TABLE", soma_table)

    client = CAVEclient(datastack)
    if version is not None:
        client.materialize.version = version
    print(f"Using version {client.materialize.version}")

    if soma_table is not None:
        soma_df = client.materialize.query_table(soma_table)
        soma_fn = f"{process_dir}/{data_filename('soma', client)}"
        soma_df.to_feather(soma_fn)
        print(f"Saved soma table to {soma_fn}")

    if cell_table is not None:
        ct_df = client.materialize.query_table(cell_table)
        ct_fn = f'{process_dir}/{data_filename("cell_type", client)}'
        ct_df.to_feather(ct_fn)
        print(f"Saved cell type table to {ct_fn}")
    pass


if __name__ == "__main__":
    cli()
